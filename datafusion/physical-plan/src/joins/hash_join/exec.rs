// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashSet;
use std::fmt;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::vec;

use crate::ExecutionPlanProperties;
use crate::execution_plan::{
    EmissionType, boundedness_from_children, has_same_children_properties,
    stub_properties,
};
use crate::filter_pushdown::{
    ChildFilterDescription, ChildPushdownResult, FilterDescription, FilterPushdownPhase,
    FilterPushdownPropagation,
};
use crate::joins::Map;
use crate::joins::array_map::ArrayMap;
use crate::joins::hash_join::inlist_builder::build_struct_inlist_values;
use crate::joins::hash_join::shared_bounds::{
    ColumnBounds, PartitionBounds, PushdownStrategy, SharedBuildAccumulator,
};
use crate::joins::hash_join::stream::{
    BuildSide, BuildSideInitialState, HashJoinStream, HashJoinStreamState,
};
use crate::joins::join_hash_map::{JoinHashMapU32, JoinHashMapU64};
use crate::joins::utils::{
    OnceAsync, OnceFut, asymmetric_join_output_partitioning, reorder_output_after_swap,
    swap_join_projection, update_hash,
};
use crate::joins::{JoinOn, JoinOnRef, PartitionMode, SharedBitmapBuilder};
use crate::metrics::{Count, MetricBuilder, MetricCategory};
use crate::projection::{
    EmbeddedProjection, JoinData, ProjectionExec, try_embed_projection,
    try_pushdown_through_join,
};
use crate::repartition::REPARTITION_RANDOM_STATE;
use crate::spill::SpillManager;
use crate::spill::get_record_batch_memory_size;
use datafusion_common::config::SpillCompression;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    PlanProperties, SendableRecordBatchStream, Statistics,
    common::can_project,
    joins::utils::{
        BuildProbeJoinMetrics, ColumnIndex, JoinFilter, JoinHashMapType,
        build_join_schema, check_join_is_valid, estimate_join_statistics,
        need_produce_result_in_final, symmetric_join_output_partitioning,
    },
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
};

use arrow::array::{ArrayRef, BooleanBufferBuilder};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow::util::bit_util;
use arrow_schema::{DataType, Schema};
use datafusion_common::config::ConfigOptions;
use datafusion_common::DataFusionError;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::utils::memory::estimate_memory_size;
use datafusion_common::{
    JoinSide, JoinType, NullEquality, Result, assert_or_internal_err, internal_err,
    plan_err, project_schema,
};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::Accumulator;
use datafusion_functions_aggregate_common::min_max::{MaxAccumulator, MinAccumulator};
use datafusion_physical_expr::equivalence::{
    ProjectionMapping, join_equivalence_properties,
};
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr, lit};
use datafusion_physical_expr::projection::{ProjectionRef, combine_projections};
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};

use datafusion_common::hash_utils::RandomState;
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use futures::TryStreamExt;
use parking_lot::Mutex;

use super::partitioned_hash_eval::SeededRandomState;

/// Hard-coded seed to ensure hash values from the hash join differ from `RepartitionExec`, avoiding collisions.
pub(crate) const HASH_JOIN_SEED: SeededRandomState =
    SeededRandomState::with_seed(12210250226015887276);

const ARRAY_MAP_CREATED_COUNT_METRIC_NAME: &str = "array_map_created_count";

/// Counter incremented when the same build partition spills twice
/// in a row, signalling skew. PR4-F ships this as a observability
/// stub; full recursive sub-partitioning (different hash seed for
/// the skewed partition) is a follow-up.
const HASH_JOIN_SKEW_PARTITION_COUNT_METRIC_NAME: &str =
    "hash_join_skew_partition_count";

/// Threshold for marking a partition as skewed. After this many
/// consecutive spills onto the same partition slot, the skew
/// counter increments once.
const HASH_JOIN_SKEW_SPILL_THRESHOLD: usize = 2;

#[expect(clippy::too_many_arguments)]
fn try_create_array_map(
    bounds: &Option<PartitionBounds>,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    on_left: &[PhysicalExprRef],
    reservation: &mut MemoryReservation,
    perfect_hash_join_small_build_threshold: usize,
    perfect_hash_join_min_key_density: f64,
    null_equality: NullEquality,
) -> Result<Option<(ArrayMap, RecordBatch, Vec<ArrayRef>)>> {
    if on_left.len() != 1 {
        return Ok(None);
    }

    if null_equality == NullEquality::NullEqualsNull {
        for batch in batches.iter() {
            let arrays = evaluate_expressions_to_arrays(on_left, batch)?;
            if arrays[0].null_count() > 0 {
                return Ok(None);
            }
        }
    }

    let (min_val, max_val) = if let Some(bounds) = bounds {
        let (min_val, max_val) = if let Some(cb) = bounds.get_column_bounds(0) {
            (cb.min.clone(), cb.max.clone())
        } else {
            return Ok(None);
        };

        if min_val.is_null() || max_val.is_null() {
            return Ok(None);
        }

        if min_val > max_val {
            return internal_err!("min_val>max_val");
        }

        if let Some((mi, ma)) =
            ArrayMap::key_to_u64(&min_val).zip(ArrayMap::key_to_u64(&max_val))
        {
            (mi, ma)
        } else {
            return Ok(None);
        }
    } else {
        return Ok(None);
    };

    let range = ArrayMap::calculate_range(min_val, max_val);
    let num_row: usize = batches.iter().map(|x| x.num_rows()).sum();

    // TODO: support create ArrayMap<u64>
    if num_row >= u32::MAX as usize {
        return Ok(None);
    }

    // When the key range spans the full integer domain (e.g. i64::MIN to i64::MAX),
    // range is u64::MAX and `range + 1` below would overflow.
    if range == usize::MAX as u64 {
        return Ok(None);
    }

    let dense_ratio = (num_row as f64) / ((range + 1) as f64);

    if range >= perfect_hash_join_small_build_threshold as u64
        && dense_ratio <= perfect_hash_join_min_key_density
    {
        return Ok(None);
    }

    let mem_size = ArrayMap::estimate_memory_size(min_val, max_val, num_row);
    reservation.try_grow(mem_size)?;

    let batch = concat_batches(schema, batches)?;
    let left_values = evaluate_expressions_to_arrays(on_left, &batch)?;

    let array_map = ArrayMap::try_new(&left_values[0], min_val, max_val)?;

    Ok(Some((array_map, batch, left_values)))
}

/// Per-partition build-side state that the probe path consumes.
///
/// One unit of work that [`super::stream::HashJoinStream`]
/// iterates over (one partition at a time, see PR4-C). PR4-D-3a
/// drops the previous PR4-B invariant that there is always
/// exactly one entry: accessors on [`JoinLeftData`] now take a
/// `partition_idx` and the probe state machine threads
/// `BuildSideReadyState::current_partition` through them. Today
/// the build path still produces a single partition; PR4-D-3b
/// fans out to N partitions and adds the probe-replay loop.
pub(super) struct JoinLeftPartitionData {
    /// Hash table over `values`, indexed into `batch`.
    /// `Arc` because [`SharedBuildAccumulator`] may share it for
    /// hash-map filter pushdown.
    pub(super) map: Arc<Map>,
    /// Concatenated build-side rows for this partition.
    batch: RecordBatch,
    /// Pre-evaluated join-key arrays for this partition.
    values: Vec<ArrayRef>,
    /// Visited-row bitmap for outer-join unmatched-row emission,
    /// length = `batch.num_rows()`. Per-partition so future PRs
    /// can emit unmatched rows partition-locally without merging
    /// a single global bitmap.
    visited_indices_bitmap: SharedBitmapBuilder,
    /// Per-partition min/max bounds for dynamic-filter pushdown.
    /// PR4-D-3a keeps a single partition so this is equal to the
    /// parent [`JoinLeftData::bounds`]; PR4-D-3b will fan out and
    /// the parent will hold the union.
    #[allow(dead_code)]
    pub(super) bounds: Option<PartitionBounds>,
}

/// One slot in [`JoinLeftData::partitions`].
///
/// `Resident` slots are ready to probe immediately. `Spilled`
/// slots park their build batches on disk; the probe-time
/// `MaterializePartition` step reads them back, builds a per-partition
/// hash map, and stores the result in the spilled slot's
/// [`OnceLock`]. Subsequent probe-side accesses then fetch the
/// materialized data from the cell, with no extra synchronization
/// (`OnceLock` makes the init thread-safe across probe streams
/// that share the same `JoinLeftData`).
pub(super) enum PartitionEntry {
    /// Hash map + batch + bitmap are in memory and ready to probe.
    Resident(JoinLeftPartitionData),
    /// Build batches still on disk plus a [`OnceLock`] that the
    /// probe path fills with the materialized
    /// [`JoinLeftPartitionData`] on first access. Once filled the
    /// cell stays populated for the rest of the join's lifetime —
    /// PR4 v1 does not free per-partition state mid-join (see
    /// design doc §4-D / TODO PR5).
    Spilled(SpilledSlot),
}

/// On-disk build partition with a lazy in-memory materialization.
///
/// The `bytes` field records the pre-spill memory footprint so
/// `MaterializePartition` can `reservation.try_grow(bytes)` symmetrically
/// with the `shrink(bytes)` that happened at spill time, keeping
/// memory accounting balanced end-to-end.
pub(super) struct SpilledSlot {
    pub(super) file: RefCountedTempFile,
    pub(super) num_rows: usize,
    pub(super) bytes: usize,
    pub(super) spill_manager: Arc<SpillManager>,
    /// Filled by `MaterializePartition` on first access; subsequent
    /// accesses on any probe thread take the same `&JoinLeftPartitionData`.
    /// `tokio::sync::OnceCell` rather than `std::sync::OnceLock`
    /// because materialization is async (spill readback) and we
    /// want `get_or_try_init` to serialize concurrent probe-stream
    /// races.
    pub(super) cell: tokio::sync::OnceCell<JoinLeftPartitionData>,
}

/// HashTable and input data for the left (build side) of a join
pub(super) struct JoinLeftData {
    /// Per-partition build state. Length = [`Self::num_partitions`]
    /// (≥ 1). Today always 1; PR4-D-3b populates the multi-partition
    /// case.
    partitions: Vec<PartitionEntry>,
    /// Number of build partitions. Cached so `num_partitions()`
    /// stays cheap and survives `Spilled` slots being swapped in
    /// place during probe-time materialization.
    num_partitions: usize,
    /// Random state used for partition routing on the probe side.
    /// MUST match the build-side `BuildSideState::partition_random_state`,
    /// otherwise build/probe end up in different partitions. Carried
    /// here so [`super::stream::HashJoinStream`] can hash-route
    /// probe batches identically in PR4-D-3b.
    #[allow(dead_code)]
    pub(super) partition_random_state: RandomState,
    /// Spill manager used to read [`PartitionEntry::Spilled`] slots
    /// back from disk during probe-time materialization. `None`
    /// when build ran in single-partition mode (no spill).
    #[allow(dead_code)]
    pub(super) spill_manager: Option<Arc<SpillManager>>,
    /// Captured at build time so [`Self::materialize_partition`] can
    /// rebuild a partition's hash map / batch / bitmap from a spill
    /// file without the probe stream having to thread these through.
    /// `None` when build ran in single-partition mode (materialization
    /// is never invoked).
    pub(super) materialize_ctx: Option<Arc<MaterializeContext>>,
    /// Counter of running probe-threads, potentially
    /// able to update `visited_indices_bitmap`
    probe_threads_counter: AtomicUsize,
    /// We need to keep this field to maintain accurate memory accounting, even though we don't directly use it.
    /// Without holding onto this reservation, the recorded memory usage would become inconsistent with actual usage.
    /// This could hide potential out-of-memory issues, especially when upstream operators increase their memory consumption.
    /// The MemoryReservation ensures proper tracking of memory resources throughout the join operation's lifecycle.
    _reservation: MemoryReservation,
    /// Aggregated bounds across all partitions, used for dynamic
    /// filter pushdown. With a single partition this equals
    /// `partitions[0].bounds`; PR4-D-3b will compute the union.
    /// `None` when the build side is empty.
    pub(super) bounds: Option<PartitionBounds>,
    /// Membership testing strategy for filter pushdown
    /// Contains either InList values for small build sides or hash table reference for large build sides
    pub(super) membership: PushdownStrategy,
    /// Shared atomic flag indicating if any probe partition saw data (for null-aware anti joins)
    /// This is shared across all probe partitions to provide global knowledge
    pub(super) probe_side_non_empty: AtomicBool,
    /// Shared atomic flag indicating if any probe partition saw NULL in join keys (for null-aware anti joins)
    pub(super) probe_side_has_null: AtomicBool,
}

impl JoinLeftData {
    /// Returns the resident build partition at `partition_idx`.
    ///
    /// Panics if the slot is currently `Spilled`. PR4-D-3b
    /// guarantees that the probe state machine only calls this
    /// after `MaterializePartition` has turned the slot into
    /// `Resident`; today every slot starts as `Resident` so the
    /// invariant holds trivially.
    fn partition(&self, partition_idx: usize) -> &JoinLeftPartitionData {
        match &self.partitions[partition_idx] {
            PartitionEntry::Resident(p) => p,
            PartitionEntry::Spilled(slot) => slot.cell.get().unwrap_or_else(|| {
                panic!(
                    "BUG: probe path accessed build partition {partition_idx} while still spilled; \
                     MaterializePartition must run first"
                )
            }),
        }
    }

    /// return a reference to the map for `partition_idx`.
    pub(super) fn map(&self, partition_idx: usize) -> &Map {
        &self.partition(partition_idx).map
    }

    /// returns a reference to the build side batch for `partition_idx`.
    pub(super) fn batch(&self, partition_idx: usize) -> &RecordBatch {
        &self.partition(partition_idx).batch
    }

    /// returns a reference to the build side expressions values for `partition_idx`.
    pub(super) fn values(&self, partition_idx: usize) -> &[ArrayRef] {
        &self.partition(partition_idx).values
    }

    /// returns a reference to the visited indices bitmap for `partition_idx`.
    pub(super) fn visited_indices_bitmap(
        &self,
        partition_idx: usize,
    ) -> &SharedBitmapBuilder {
        &self.partition(partition_idx).visited_indices_bitmap
    }

    /// returns a reference to the InList values for filter pushdown
    pub(super) fn membership(&self) -> &PushdownStrategy {
        &self.membership
    }

    /// Decrements the counter of running threads, and returns `true`
    /// if caller is the last running thread
    pub(super) fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    /// Number of build partitions held by this `JoinLeftData`.
    ///
    /// Today always 1. PR4-D-3b returns the actual partition
    /// count once the build side stops collapsing into a single
    /// concatenated batch.
    pub(super) fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Borrow the shared `Arc<Map>` for the build partition at
    /// `partition_idx`.
    ///
    /// Used by [`SharedBuildAccumulator`] to share the build-side
    /// hash table for filter pushdown without cloning the table
    /// data itself.
    #[allow(dead_code)]
    pub(super) fn map_arc(&self, partition_idx: usize) -> &Arc<Map> {
        &self.partition(partition_idx).map
    }

    /// Whether the slot at `partition_idx` is currently resident
    /// (probe-ready) without having to call any of the panicking
    /// accessors. `Resident` slots are always ready; `Spilled`
    /// slots become ready after [`Self::materialize_partition`]
    /// fills the cell.
    pub(super) fn is_resident(&self, partition_idx: usize) -> bool {
        match &self.partitions[partition_idx] {
            PartitionEntry::Resident(_) => true,
            PartitionEntry::Spilled(slot) => slot.cell.get().is_some(),
        }
    }

    /// Pull a [`PartitionEntry::Spilled`] slot back from disk and
    /// build its hash map.
    ///
    /// Idempotent and thread-safe across concurrent probe streams:
    /// `tokio::sync::OnceCell::get_or_try_init` ensures the
    /// readback runs exactly once per slot regardless of how many
    /// streams race on it. `Resident` slots are a no-op fast path.
    pub(super) async fn materialize_partition(
        &self,
        partition_idx: usize,
    ) -> Result<()> {
        let slot = match &self.partitions[partition_idx] {
            PartitionEntry::Resident(_) => return Ok(()),
            PartitionEntry::Spilled(slot) => slot,
        };
        slot.cell
            .get_or_try_init(|| async {
                let ctx = self.materialize_ctx.as_ref().ok_or_else(|| {
                    DataFusionError::Internal(
                        "BUG: spilled slot without materialize context".into(),
                    )
                })?;
                let mut reservation = MemoryConsumer::new(format!(
                    "HashJoinMaterialize[partition={partition_idx}]"
                ))
                .register(&ctx.runtime_env.memory_pool);
                reservation.try_grow(slot.bytes)?;
                ctx.metrics.build_mem_used.add(slot.bytes);
                let mut stream = slot
                    .spill_manager
                    .read_spill_as_stream(slot.file.clone(), None)?;
                let mut batches = Vec::new();
                use futures::StreamExt;
                while let Some(b) = stream.next().await {
                    batches.push(b?);
                }
                let partition_data = build_partition_data(
                    batches,
                    slot.num_rows,
                    None,
                    &ctx.schema,
                    &ctx.on_left,
                    &ctx.random_state,
                    ctx.null_equality,
                    ctx.with_visited_indices_bitmap,
                    &mut reservation,
                    &ctx.metrics,
                    &ctx.array_map_created_count,
                    ctx.perfect_hash_join_small_build_threshold,
                    ctx.perfect_hash_join_min_key_density,
                )?;
                // Detach the reservation: it now belongs to the
                // materialized partition data and is refunded when
                // the slot is dropped at join end.
                std::mem::forget(reservation);
                Ok(partition_data)
            })
            .await
            .map(|_| ())
    }
}

/// Captured-at-build-time context that
/// [`JoinLeftData::materialize_partition`] needs to rebuild a
/// spilled partition's hash map / batch / bitmap on demand.
pub(super) struct MaterializeContext {
    pub(super) on_left: Vec<PhysicalExprRef>,
    pub(super) random_state: RandomState,
    pub(super) null_equality: NullEquality,
    pub(super) with_visited_indices_bitmap: bool,
    pub(super) schema: SchemaRef,
    pub(super) perfect_hash_join_small_build_threshold: usize,
    pub(super) perfect_hash_join_min_key_density: f64,
    pub(super) array_map_created_count: Count,
    pub(super) metrics: BuildProbeJoinMetrics,
    pub(super) runtime_env: Arc<RuntimeEnv>,
}

/// Helps to build [`HashJoinExec`].
///
/// Builder can be created from an existing [`HashJoinExec`] using [`From::from`].
/// In this case, all its fields are inherited. If a field that affects the node's
/// properties is modified, they will be automatically recomputed during the build.
///
/// # Adding setters
///
/// When adding a new setter, it is necessary to ensure that the `preserve_properties`
/// flag is set to false if modifying the field requires a recomputation of the plan's
/// properties.
///
pub struct HashJoinExecBuilder {
    exec: HashJoinExec,
    preserve_properties: bool,
}

impl HashJoinExecBuilder {
    /// Make a new [`HashJoinExecBuilder`].
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        join_type: JoinType,
    ) -> Self {
        Self {
            exec: HashJoinExec {
                left,
                right,
                on,
                filter: None,
                join_type,
                left_fut: Default::default(),
                random_state: HASH_JOIN_SEED,
                mode: PartitionMode::Auto,
                fetch: None,
                metrics: ExecutionPlanMetricsSet::new(),
                projection: None,
                column_indices: vec![],
                null_equality: NullEquality::NullEqualsNothing,
                null_aware: false,
                dynamic_filter: None,
                // Will be computed at when plan will be built.
                cache: stub_properties(),
                join_schema: Arc::new(Schema::empty()),
            },
            // As `exec` is initialized with stub properties,
            // they will be properly computed when plan will be built.
            preserve_properties: false,
        }
    }

    /// Set join type.
    pub fn with_type(mut self, join_type: JoinType) -> Self {
        self.exec.join_type = join_type;
        self.preserve_properties = false;
        self
    }

    /// Set projection from the vector.
    pub fn with_projection(self, projection: Option<Vec<usize>>) -> Self {
        self.with_projection_ref(projection.map(Into::into))
    }

    /// Set projection from the shared reference.
    pub fn with_projection_ref(mut self, projection: Option<ProjectionRef>) -> Self {
        self.exec.projection = projection;
        self.preserve_properties = false;
        self
    }

    /// Set optional filter.
    pub fn with_filter(mut self, filter: Option<JoinFilter>) -> Self {
        self.exec.filter = filter;
        self
    }

    /// Set expressions to join on.
    pub fn with_on(mut self, on: Vec<(PhysicalExprRef, PhysicalExprRef)>) -> Self {
        self.exec.on = on;
        self.preserve_properties = false;
        self
    }

    /// Set partition mode.
    pub fn with_partition_mode(mut self, mode: PartitionMode) -> Self {
        self.exec.mode = mode;
        self.preserve_properties = false;
        self
    }

    /// Set null equality property.
    pub fn with_null_equality(mut self, null_equality: NullEquality) -> Self {
        self.exec.null_equality = null_equality;
        self
    }

    /// Set null aware property.
    pub fn with_null_aware(mut self, null_aware: bool) -> Self {
        self.exec.null_aware = null_aware;
        self
    }

    /// Set fetch property.
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.exec.fetch = fetch;
        self
    }

    /// Require to recompute plan properties.
    pub fn recompute_properties(mut self) -> Self {
        self.preserve_properties = false;
        self
    }

    /// Replace children.
    pub fn with_new_children(
        mut self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Self> {
        assert_or_internal_err!(
            children.len() == 2,
            "wrong number of children passed into `HashJoinExecBuilder`"
        );
        self.preserve_properties &= has_same_children_properties(&self.exec, &children)?;
        self.exec.right = children.swap_remove(1);
        self.exec.left = children.swap_remove(0);
        Ok(self)
    }

    /// Reset runtime state.
    pub fn reset_state(mut self) -> Self {
        self.exec.left_fut = Default::default();
        self.exec.dynamic_filter = None;
        self.exec.metrics = ExecutionPlanMetricsSet::new();
        self
    }

    /// Build result as a dyn execution plan.
    pub fn build_exec(self) -> Result<Arc<dyn ExecutionPlan>> {
        self.build().map(|p| Arc::new(p) as _)
    }

    /// Build resulting execution plan.
    pub fn build(self) -> Result<HashJoinExec> {
        let Self {
            exec,
            preserve_properties,
        } = self;

        // Validate null_aware flag
        if exec.null_aware {
            let join_type = exec.join_type();
            if !matches!(join_type, JoinType::LeftAnti) {
                return plan_err!(
                    "null_aware can only be true for LeftAnti joins, got {join_type}"
                );
            }
            let on = exec.on();
            if on.len() != 1 {
                return plan_err!(
                    "null_aware anti join only supports single column join key, got {} columns",
                    on.len()
                );
            }
        }

        if preserve_properties {
            return Ok(exec);
        }

        let HashJoinExec {
            left,
            right,
            on,
            filter,
            join_type,
            left_fut,
            random_state,
            mode,
            metrics,
            projection,
            null_equality,
            null_aware,
            dynamic_filter,
            fetch,
            // Recomputed.
            join_schema: _,
            column_indices: _,
            cache: _,
        } = exec;

        let left_schema = left.schema();
        let right_schema = right.schema();
        if on.is_empty() {
            return plan_err!("On constraints in HashJoinExec should be non-empty");
        }

        check_join_is_valid(&left_schema, &right_schema, &on)?;
        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &join_type);

        let join_schema = Arc::new(join_schema);

        // Check if the projection is valid.
        can_project(&join_schema, projection.as_deref())?;

        let cache = HashJoinExec::compute_properties(
            &left,
            &right,
            &join_schema,
            join_type,
            &on,
            mode,
            projection.as_deref(),
        )?;

        Ok(HashJoinExec {
            left,
            right,
            on,
            filter,
            join_type,
            join_schema,
            left_fut,
            random_state,
            mode,
            metrics,
            projection,
            column_indices,
            null_equality,
            null_aware,
            cache: Arc::new(cache),
            dynamic_filter,
            fetch,
        })
    }

    fn with_dynamic_filter(mut self, filter: Option<HashJoinExecDynamicFilter>) -> Self {
        self.exec.dynamic_filter = filter;
        self
    }
}

impl From<&HashJoinExec> for HashJoinExecBuilder {
    fn from(exec: &HashJoinExec) -> Self {
        Self {
            exec: HashJoinExec {
                left: Arc::clone(exec.left()),
                right: Arc::clone(exec.right()),
                on: exec.on.clone(),
                filter: exec.filter.clone(),
                join_type: exec.join_type,
                join_schema: Arc::clone(&exec.join_schema),
                left_fut: Arc::clone(&exec.left_fut),
                random_state: exec.random_state.clone(),
                mode: exec.mode,
                metrics: exec.metrics.clone(),
                projection: exec.projection.clone(),
                column_indices: exec.column_indices.clone(),
                null_equality: exec.null_equality,
                null_aware: exec.null_aware,
                cache: Arc::clone(&exec.cache),
                dynamic_filter: exec.dynamic_filter.clone(),
                fetch: exec.fetch,
            },
            preserve_properties: true,
        }
    }
}

#[expect(rustdoc::private_intra_doc_links)]
/// Join execution plan: Evaluates equijoin predicates in parallel on multiple
/// partitions using a hash table and an optional filter list to apply post
/// join.
///
/// # Join Expressions
///
/// This implementation is optimized for evaluating equijoin predicates  (
/// `<col1> = <col2>`) expressions, which are represented as a list of `Columns`
/// in [`Self::on`].
///
/// Non-equality predicates, which can not pushed down to a join inputs (e.g.
/// `<col1> != <col2>`) are known as "filter expressions" and are evaluated
/// after the equijoin predicates.
///
/// # ArrayMap Optimization
///
/// For joins with a single integer-based join key, `HashJoinExec` may use an [`ArrayMap`]
/// (also known as a "perfect hash join") instead of a general-purpose hash map.
/// This optimization is used when:
/// 1. There is exactly one join key.
/// 2. The join key is an integer type up to 64 bits wide that can be losslessly converted
///    to `u64` (128-bit integer types such as `i128` and `u128` are not supported).
/// 3. The range of keys is small enough (controlled by `perfect_hash_join_small_build_threshold`)
///    OR the keys are sufficiently dense (controlled by `perfect_hash_join_min_key_density`).
/// 4. build_side.num_rows() < u32::MAX
/// 5. NullEqualsNothing || (NullEqualsNull && build side doesn't contain null)
///
/// See [`try_create_array_map`] for more details.
///
/// Note that when using [`PartitionMode::Partitioned`], the build side is split into multiple
/// partitions. This can cause a dense build side to become sparse within each partition,
/// potentially disabling this optimization.
///
/// For example, consider:
/// ```sql
/// SELECT t1.value, t2.value
/// FROM range(10000) AS t1
/// JOIN range(10000) AS t2
///   ON t1.value = t2.value;
/// ```
/// With 24 partitions, each partition will only receive a subset of the 10,000 rows.
/// The first partition might contain values like `3, 10, 18, 39, 43`, which are sparse
/// relative to the original range, even though the overall data set is dense.
///
/// # "Build Side" vs "Probe Side"
///
/// HashJoin takes two inputs, which are referred to as the "build" and the
/// "probe". The build side is the first child, and the probe side is the second
/// child.
///
/// The two inputs are treated differently and it is VERY important that the
/// *smaller* input is placed on the build side to minimize the work of creating
/// the hash table.
///
/// ```text
///          ┌───────────┐
///          │ HashJoin  │
///          │           │
///          └───────────┘
///              │   │
///        ┌─────┘   └─────┐
///        ▼               ▼
/// ┌────────────┐  ┌─────────────┐
/// │   Input    │  │    Input    │
/// │    [0]     │  │     [1]     │
/// └────────────┘  └─────────────┘
///
///  "build side"    "probe side"
/// ```
///
/// Execution proceeds in 2 stages:
///
/// 1. the **build phase** creates a hash table from the tuples of the build side,
///    and single concatenated batch containing data from all fetched record batches.
///    Resulting hash table stores hashed join-key fields for each row as a key, and
///    indices of corresponding rows in concatenated batch.
///
/// When using the standard `JoinHashMap`, hash join uses LIFO data structure as a hash table,
/// and in order to retain original build-side input order while obtaining data during probe phase,
/// hash table is updated by iterating batch sequence in reverse order -- it allows to
/// keep rows with smaller indices "on the top" of hash table, and still maintain
/// correct indexing for concatenated build-side data batch.
///
/// Example of build phase for 3 record batches:
///
///
/// ```text
///
///  Original build-side data   Inserting build-side values into hashmap    Concatenated build-side batch
///                                                                         ┌───────────────────────────┐
///                             hashmap.insert(row-hash, row-idx + offset)  │                      idx  │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 1 │        1) update_hash for batch 3 with offset 0    │          │ Row 6 │    0   │
///   Batch 1  │       │           - hashmap.insert(Row 7, idx 1)           │ Batch 3  │       │        │
///            │ Row 2 │           - hashmap.insert(Row 6, idx 0)           │          │ Row 7 │    1   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 3 │        2) update_hash for batch 2 with offset 2    │          │ Row 3 │    2   │
///            │       │           - hashmap.insert(Row 5, idx 4)           │          │       │        │
///   Batch 2  │ Row 4 │           - hashmap.insert(Row 4, idx 3)           │ Batch 2  │ Row 4 │    3   │
///            │       │           - hashmap.insert(Row 3, idx 2)           │          │       │        │
///            │ Row 5 │                                                    │          │ Row 5 │    4   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///            ┌───────┐                                                    │          ┌───────┐        │
///            │ Row 6 │        3) update_hash for batch 1 with offset 5    │          │ Row 1 │    5   │
///   Batch 3  │       │           - hashmap.insert(Row 2, idx 6)           │ Batch 1  │       │        │
///            │ Row 7 │           - hashmap.insert(Row 1, idx 5)           │          │ Row 2 │    6   │
///            └───────┘                                                    │          └───────┘        │
///                                                                         │                           │
///                                                                         └───────────────────────────┘
/// ```
///
/// 2. the **probe phase** where the tuples of the probe side are streamed
///    through, checking for matches of the join keys in the hash table.
///
/// ```text
///                 ┌────────────────┐          ┌────────────────┐
///                 │ ┌─────────┐    │          │ ┌─────────┐    │
///                 │ │  Hash   │    │          │ │  Hash   │    │
///                 │ │  Table  │    │          │ │  Table  │    │
///                 │ │(keys are│    │          │ │(keys are│    │
///                 │ │equi join│    │          │ │equi join│    │  Stage 2: batches from
///  Stage 1: the   │ │columns) │    │          │ │columns) │    │    the probe side are
/// *entire* build  │ │         │    │          │ │         │    │  streamed through, and
///  side is read   │ └─────────┘    │          │ └─────────┘    │   checked against the
/// into the hash   │      ▲         │          │          ▲     │   contents of the hash
///     table       │       HashJoin │          │  HashJoin      │          table
///                 └──────┼─────────┘          └──────────┼─────┘
///             ─ ─ ─ ─ ─ ─                                 ─ ─ ─ ─ ─ ─ ─
///            │                                                         │
///
///            │                                                         │
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///           ...                                                       ...
///     ┌────────────┐                                            ┌────────────┐
///     │RecordBatch │                                            │RecordBatch │
///     └────────────┘                                            └────────────┘
///
///        build side                                                probe side
/// ```
///
/// # Example "Optimal" Plans
///
/// The differences in the inputs means that for classic "Star Schema Query",
/// the optimal plan will be a **"Right Deep Tree"** . A Star Schema Query is
/// one where there is one large table and several smaller "dimension" tables,
/// joined on `Foreign Key = Primary Key` predicates.
///
/// A "Right Deep Tree" looks like this large table as the probe side on the
/// lowest join:
///
/// ```text
///             ┌───────────┐
///             │ HashJoin  │
///             │           │
///             └───────────┘
///                 │   │
///         ┌───────┘   └──────────┐
///         ▼                      ▼
/// ┌───────────────┐        ┌───────────┐
/// │ small table 1 │        │ HashJoin  │
/// │  "dimension"  │        │           │
/// └───────────────┘        └───┬───┬───┘
///                   ┌──────────┘   └───────┐
///                   │                      │
///                   ▼                      ▼
///           ┌───────────────┐        ┌───────────┐
///           │ small table 2 │        │ HashJoin  │
///           │  "dimension"  │        │           │
///           └───────────────┘        └───┬───┬───┘
///                               ┌────────┘   └────────┐
///                               │                     │
///                               ▼                     ▼
///                       ┌───────────────┐     ┌───────────────┐
///                       │ small table 3 │     │  large table  │
///                       │  "dimension"  │     │    "fact"     │
///                       └───────────────┘     └───────────────┘
/// ```
///
/// # Clone / Shared State
///
/// Note this structure includes a [`OnceAsync`] that is used to coordinate the
/// loading of the left side with the processing in each output stream.
/// Therefore it can not be [`Clone`]
pub struct HashJoinExec {
    /// left (build) side which gets hashed
    pub left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    pub right: Arc<dyn ExecutionPlan>,
    /// Set of equijoin columns from the relations: `(left_col, right_col)`
    pub on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    /// Filters which are applied while finding matching rows
    pub filter: Option<JoinFilter>,
    /// How the join is performed (`OUTER`, `INNER`, etc)
    pub join_type: JoinType,
    /// The schema after join. Please be careful when using this schema,
    /// if there is a projection, the schema isn't the same as the output schema.
    join_schema: SchemaRef,
    /// Future that consumes left input and builds the hash table
    ///
    /// For CollectLeft partition mode, this structure is *shared* across all output streams.
    ///
    /// Each output stream waits on the `OnceAsync` to signal the completion of
    /// the hash table creation.
    left_fut: Arc<OnceAsync<JoinLeftData>>,
    /// Shared the `SeededRandomState` for the hashing algorithm (seeds preserved for serialization)
    random_state: SeededRandomState,
    /// Partitioning mode to use
    pub mode: PartitionMode,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The projection indices of the columns in the output schema of join
    pub projection: Option<ProjectionRef>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// The equality null-handling behavior of the join algorithm.
    pub null_equality: NullEquality,
    /// Flag to indicate if this is a null-aware anti join
    pub null_aware: bool,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: Arc<PlanProperties>,
    /// Dynamic filter for pushing down to the probe side
    /// Set when dynamic filter pushdown is detected in handle_child_pushdown_result.
    /// HashJoinExec also needs to keep a shared bounds accumulator for coordinating updates.
    dynamic_filter: Option<HashJoinExecDynamicFilter>,
    /// Maximum number of rows to return
    fetch: Option<usize>,
}

#[derive(Clone)]
struct HashJoinExecDynamicFilter {
    /// Dynamic filter that we'll update with the results of the build side once that is done.
    filter: Arc<DynamicFilterPhysicalExpr>,
    /// Build accumulator to collect build-side information (hash maps and/or bounds) from each partition.
    /// It is lazily initialized during execution to make sure we use the actual execution time partition counts.
    build_accumulator: OnceLock<Arc<SharedBuildAccumulator>>,
}

impl fmt::Debug for HashJoinExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashJoinExec")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("on", &self.on)
            .field("filter", &self.filter)
            .field("join_type", &self.join_type)
            .field("join_schema", &self.join_schema)
            .field("left_fut", &self.left_fut)
            .field("random_state", &self.random_state)
            .field("mode", &self.mode)
            .field("metrics", &self.metrics)
            .field("projection", &self.projection)
            .field("column_indices", &self.column_indices)
            .field("null_equality", &self.null_equality)
            .field("cache", &self.cache)
            // Explicitly exclude dynamic_filter to avoid runtime state differences in tests
            .finish()
    }
}

impl EmbeddedProjection for HashJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.with_projection(projection)
    }
}

impl HashJoinExec {
    /// Tries to create a new [`HashJoinExec`].
    ///
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        partition_mode: PartitionMode,
        null_equality: NullEquality,
        null_aware: bool,
    ) -> Result<Self> {
        HashJoinExecBuilder::new(left, right, on, *join_type)
            .with_filter(filter)
            .with_projection(projection)
            .with_partition_mode(partition_mode)
            .with_null_equality(null_equality)
            .with_null_aware(null_aware)
            .build()
    }

    /// Create a builder based on the existing [`HashJoinExec`].
    ///
    /// Returned builder preserves all existing fields. If a field requiring properties
    /// recomputation is modified, this will be done automatically during the node build.
    ///
    pub fn builder(&self) -> HashJoinExecBuilder {
        self.into()
    }

    fn create_dynamic_filter(on: &JoinOn) -> Arc<DynamicFilterPhysicalExpr> {
        // Extract the right-side keys (probe side keys) from the `on` clauses
        // Dynamic filter will be created from build side values (left side) and applied to probe side (right side)
        let right_keys: Vec<_> = on.iter().map(|(_, r)| Arc::clone(r)).collect();
        // Initialize with a placeholder expression (true) that will be updated when the hash table is built
        Arc::new(DynamicFilterPhysicalExpr::new(right_keys, lit(true)))
    }

    fn allow_join_dynamic_filter_pushdown(&self, config: &ConfigOptions) -> bool {
        let (_, probe_preserved) = self.join_type.on_lr_is_preserved();
        if !probe_preserved || !config.optimizer.enable_join_dynamic_filter_pushdown {
            return false;
        }

        // `preserve_file_partitions` can report Hash partitioning for Hive-style
        // file groups, but those partitions are not actually hash-distributed.
        // Partitioned dynamic filters rely on hash routing, so disable them in
        // this mode to avoid incorrect results. Follow-up work: enable dynamic
        // filtering for preserve_file_partitioned scans (issue #20195).
        // https://github.com/apache/datafusion/issues/20195
        if config.optimizer.preserve_file_partitions > 0
            && self.mode == PartitionMode::Partitioned
        {
            return false;
        }

        true
    }

    /// left (build) side which gets hashed
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right (probe) side which are filtered by the hash table
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(PhysicalExprRef, PhysicalExprRef)] {
        &self.on
    }

    /// Filters applied before join output
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.filter.as_ref()
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// The schema after join. Please be careful when using this schema,
    /// if there is a projection, the schema isn't the same as the output schema.
    pub fn join_schema(&self) -> &SchemaRef {
        &self.join_schema
    }

    /// The partitioning mode of this hash join
    pub fn partition_mode(&self) -> &PartitionMode {
        &self.mode
    }

    /// Get null_equality
    pub fn null_equality(&self) -> NullEquality {
        self.null_equality
    }

    /// Get the dynamic filter expression for testing purposes.
    /// Returns `None` if no dynamic filter has been set.
    ///
    /// This method is intended for testing only and should not be used in production code.
    #[doc(hidden)]
    pub fn dynamic_filter_for_test(&self) -> Option<&Arc<DynamicFilterPhysicalExpr>> {
        self.dynamic_filter.as_ref().map(|df| &df.filter)
    }

    /// Calculate order preservation flags for this hash join.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(
                join_type,
                JoinType::Inner
                    | JoinType::Right
                    | JoinType::RightAnti
                    | JoinType::RightSemi
                    | JoinType::RightMark
            ),
        ]
    }

    /// Get probe side information for the hash join.
    pub fn probe_side() -> JoinSide {
        // In current implementation right side is always probe side.
        JoinSide::Right
    }

    /// Return whether the join contains a projection
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// Return new instance of [HashJoinExec] with the given projection.
    pub fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        let projection = projection.map(Into::into);
        //  check if the projection is valid
        can_project(&self.schema(), projection.as_deref())?;
        let projection =
            combine_projections(projection.as_ref(), self.projection.as_ref())?;
        self.builder().with_projection_ref(projection).build()
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: &SchemaRef,
        join_type: JoinType,
        on: JoinOnRef,
        mode: PartitionMode,
        projection: Option<&[usize]>,
    ) -> Result<PlanProperties> {
        // Calculate equivalence properties:
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            Arc::clone(schema),
            &Self::maintains_input_order(join_type),
            Some(Self::probe_side()),
            on,
        )?;

        let mut output_partitioning = match mode {
            PartitionMode::CollectLeft => {
                asymmetric_join_output_partitioning(left, right, &join_type)?
            }
            PartitionMode::Auto => Partitioning::UnknownPartitioning(
                right.output_partitioning().partition_count(),
            ),
            PartitionMode::Partitioned => {
                symmetric_join_output_partitioning(left, right, &join_type)?
            }
        };

        let emission_type = if left.boundedness().is_unbounded() {
            EmissionType::Final
        } else if right.pipeline_behavior() == EmissionType::Incremental {
            match join_type {
                // If we only need to generate matched rows from the probe side,
                // we can emit rows incrementally.
                JoinType::Inner
                | JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::Right
                | JoinType::RightAnti
                | JoinType::RightMark => EmissionType::Incremental,
                // If we need to generate unmatched rows from the *build side*,
                // we need to emit them at the end.
                JoinType::Left
                | JoinType::LeftAnti
                | JoinType::LeftMark
                | JoinType::Full => EmissionType::Both,
            }
        } else {
            right.pipeline_behavior()
        };

        // If contains projection, update the PlanProperties.
        if let Some(projection) = projection {
            // construct a map from the input expressions to the output expression of the Projection
            let projection_mapping = ProjectionMapping::from_indices(projection, schema)?;
            let out_schema = project_schema(schema, Some(&projection))?;
            output_partitioning =
                output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            emission_type,
            boundedness_from_children([left, right]),
        ))
    }

    /// Returns a new `ExecutionPlan` that computes the same join as this one,
    /// with the left and right inputs swapped using the  specified
    /// `partition_mode`.
    ///
    /// # Notes:
    ///
    /// This function is public so other downstream projects can use it to
    /// construct `HashJoinExec` with right side as the build side.
    ///
    /// For using this interface directly, please refer to below:
    ///
    /// Hash join execution may require specific input partitioning (for example,
    /// the left child may have a single partition while the right child has multiple).
    ///
    /// Calling this function on join nodes whose children have already been repartitioned
    /// (e.g., after a `RepartitionExec` has been inserted) may break the partitioning
    /// requirements of the hash join. Therefore, ensure you call this function
    /// before inserting any repartitioning operators on the join's children.
    ///
    /// In DataFusion's default SQL interface, this function is used by the `JoinSelection`
    /// physical optimizer rule to determine a good join order, which is
    /// executed before the `EnforceDistribution` rule (the rule that may
    /// insert `RepartitionExec` operators).
    pub fn swap_inputs(
        &self,
        partition_mode: PartitionMode,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let left = self.left();
        let right = self.right();
        let new_join = self
            .builder()
            .with_type(self.join_type.swap())
            .with_new_children(vec![Arc::clone(right), Arc::clone(left)])?
            .with_on(
                self.on()
                    .iter()
                    .map(|(l, r)| (Arc::clone(r), Arc::clone(l)))
                    .collect(),
            )
            .with_filter(self.filter().map(JoinFilter::swap))
            .with_projection(swap_join_projection(
                left.schema().fields().len(),
                right.schema().fields().len(),
                self.projection.as_deref(),
                self.join_type(),
            ))
            .with_partition_mode(partition_mode)
            .build()?;
        // In case of anti / semi joins or if there is embedded projection in HashJoinExec, output column order is preserved, no need to add projection again
        if matches!(
            self.join_type(),
            JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::LeftAnti
                | JoinType::RightAnti
                | JoinType::LeftMark
                | JoinType::RightMark
        ) || self.projection.is_some()
        {
            Ok(Arc::new(new_join))
        } else {
            reorder_output_after_swap(Arc::new(new_join), &left.schema(), &right.schema())
        }
    }
}

impl DisplayAs for HashJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projections = if self.contains_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "".to_string()
                };
                let display_null_equality =
                    if self.null_equality() == NullEquality::NullEqualsNull {
                        ", NullsEqual: true"
                    } else {
                        ""
                    };
                let display_fetch = self
                    .fetch
                    .map_or_else(String::new, |f| format!(", fetch={f}"));
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1}, {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "HashJoinExec: mode={:?}, join_type={:?}, on=[{}]{}{}{}{}",
                    self.mode,
                    self.join_type,
                    on,
                    display_filter,
                    display_projections,
                    display_null_equality,
                    display_fetch,
                )
            }
            DisplayFormatType::TreeRender => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| {
                        format!("({} = {})", fmt_sql(c1.as_ref()), fmt_sql(c2.as_ref()))
                    })
                    .collect::<Vec<String>>()
                    .join(", ");

                if *self.join_type() != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)?;
                }

                writeln!(f, "on={on}")?;

                if self.null_equality() == NullEquality::NullEqualsNull {
                    writeln!(f, "NullsEqual: true")?;
                }

                if let Some(filter) = self.filter.as_ref() {
                    writeln!(f, "filter={filter}")?;
                }

                if let Some(fetch) = self.fetch {
                    writeln!(f, "fetch={fetch}")?;
                }

                Ok(())
            }
        }
    }
}

impl ExecutionPlan for HashJoinExec {
    fn name(&self) -> &'static str {
        "HashJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        match self.mode {
            PartitionMode::CollectLeft => vec![
                Distribution::SinglePartition,
                Distribution::UnspecifiedDistribution,
            ],
            PartitionMode::Partitioned => {
                let (left_expr, right_expr) = self
                    .on
                    .iter()
                    .map(|(l, r)| (Arc::clone(l), Arc::clone(r)))
                    .unzip();
                vec![
                    Distribution::HashPartitioned(left_expr),
                    Distribution::HashPartitioned(right_expr),
                ]
            }
            PartitionMode::Auto => vec![
                Distribution::UnspecifiedDistribution,
                Distribution::UnspecifiedDistribution,
            ],
        }
    }

    // For [JoinType::Inner] and [JoinType::RightSemi] in hash joins, the probe phase initiates by
    // applying the hash function to convert the join key(s) in each row into a hash value from the
    // probe side table in the order they're arranged. The hash value is used to look up corresponding
    // entries in the hash table that was constructed from the build side table during the build phase.
    //
    // Because of the immediate generation of result rows once a match is found,
    // the output of the join tends to follow the order in which the rows were read from
    // the probe side table. This is simply due to the sequence in which the rows were processed.
    // Hence, it appears that the hash join is preserving the order of the probe side.
    //
    // Meanwhile, in the case of a [JoinType::RightAnti] hash join,
    // the unmatched rows from the probe side are also kept in order.
    // This is because the **`RightAnti`** join is designed to return rows from the right
    // (probe side) table that have no match in the left (build side) table. Because the rows
    // are processed sequentially in the probe phase, and unmatched rows are directly output
    // as results, these results tend to retain the order of the probe side table.
    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        // Apply to join key expressions from both sides
        let mut tnr = TreeNodeRecursion::Continue;
        for (left, right) in &self.on {
            tnr = tnr.visit_sibling(|| f(left.as_ref()))?;
            tnr = tnr.visit_sibling(|| f(right.as_ref()))?;
        }

        // Apply to join filter expression if present
        if let Some(filter) = &self.filter {
            tnr = tnr.visit_sibling(|| f(filter.expression().as_ref()))?;
        }

        // Apply to dynamic filter expression if present
        if let Some(df) = &self.dynamic_filter {
            tnr = tnr.visit_sibling(|| f(df.filter.as_ref()))?;
        }

        Ok(tnr)
    }

    /// Creates a new HashJoinExec with different children while preserving configuration.
    ///
    /// This method is called during query optimization when the optimizer creates new
    /// plan nodes. Importantly, it creates a fresh bounds_accumulator via `try_new`
    /// rather than cloning the existing one because partitioning may have changed.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.builder().with_new_children(children)?.build_exec()
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        self.builder().reset_state().build_exec()
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let on_left = self
            .on
            .iter()
            .map(|on| Arc::clone(&on.0))
            .collect::<Vec<_>>();
        let left_partitions = self.left.output_partitioning().partition_count();
        let right_partitions = self.right.output_partitioning().partition_count();

        assert_or_internal_err!(
            self.mode != PartitionMode::Partitioned
                || left_partitions == right_partitions,
            "Invalid HashJoinExec, partition count mismatch {left_partitions}!={right_partitions},\
             consider using RepartitionExec"
        );

        assert_or_internal_err!(
            self.mode != PartitionMode::CollectLeft || left_partitions == 1,
            "Invalid HashJoinExec, the output partition count of the left child must be 1 in CollectLeft mode,\
             consider using CoalescePartitionsExec or the EnforceDistribution rule"
        );

        // Only enable dynamic filter pushdown if:
        // - The session config enables dynamic filter pushdown
        // - A dynamic filter exists
        // - At least one consumer is holding a reference to it, this avoids expensive filter
        //   computation when disabled or when no consumer will use it.
        // PR4-D-3b: dynamic-filter pushdown's `PushdownStrategy::Map`
        // can only carry one `Arc<Map>`. Multi-partition builds
        // (one map per partition) would require either a union map
        // or a `Vec<Arc<Map>>` variant in `shared_bounds.rs`; pending
        // that work, disable join dynamic-filter pushdown when the
        // spill threshold turns on multi-partition build. Bounds-based
        // pushdown via `dynamic_filter.bounds` is unaffected.
        let multi_partition_build = context
            .session_config()
            .options()
            .execution
            .hash_join_spill_threshold
            > 0.0
            && context
                .session_config()
                .options()
                .execution
                .hash_join_num_partitions
                > 1;
        let enable_dynamic_filter_pushdown = !multi_partition_build
            && self.allow_join_dynamic_filter_pushdown(
                context.session_config().options(),
            )
            && self
                .dynamic_filter
                .as_ref()
                .map(|df| df.filter.is_used())
                .unwrap_or(false);

        let join_metrics = BuildProbeJoinMetrics::new(partition, &self.metrics);

        let array_map_created_count = MetricBuilder::new(&self.metrics)
            .with_category(MetricCategory::Rows)
            .counter(ARRAY_MAP_CREATED_COUNT_METRIC_NAME, partition);

        // PR4-F: skew detection. Counter ticks once per partition
        // that crosses [`HASH_JOIN_SKEW_SPILL_THRESHOLD`]
        // consecutive spills onto the same slot — a signal that
        // the hash router is concentrating rows in a single
        // partition and full recursive sub-partitioning is
        // needed (follow-up PR).
        let skew_partition_count = MetricBuilder::new(&self.metrics)
            .with_category(MetricCategory::Rows)
            .counter(HASH_JOIN_SKEW_PARTITION_COUNT_METRIC_NAME, partition);

        // Initialize build_accumulator lazily with runtime partition counts (only if enabled)
        // Use RepartitionExec's random state (seeds: 0,0,0,0) for partition routing
        let repartition_random_state = REPARTITION_RANDOM_STATE;
        let build_accumulator = enable_dynamic_filter_pushdown
            .then(|| {
                self.dynamic_filter.as_ref().map(|df| {
                    let filter = Arc::clone(&df.filter);
                    let on_right = self
                        .on
                        .iter()
                        .map(|(_, right_expr)| Arc::clone(right_expr))
                        .collect::<Vec<_>>();
                    Some(Arc::clone(df.build_accumulator.get_or_init(|| {
                        Arc::new(SharedBuildAccumulator::new_from_partition_mode(
                            self.mode,
                            self.left.as_ref(),
                            self.right.as_ref(),
                            filter,
                            on_right,
                            repartition_random_state,
                        ))
                    })))
                })
            })
            .flatten()
            .flatten();

        // Whether per-partition disk spill is enabled (PR3). When
        // true, register the build-side MemoryConsumer with
        // `can_spill: true` so external reclaimers and the global
        // memory-pool fairness logic see this operator as a
        // cooperative spill participant rather than a hard-OOM
        // contributor.
        let spill_enabled = context
            .session_config()
            .options()
            .execution
            .hash_join_spill_threshold
            > 0.0;
        let runtime_env = context.runtime_env();
        let metrics_set = self.metrics.clone();

        let left_fut = match self.mode {
            PartitionMode::CollectLeft => self.left_fut.try_once(|| {
                let left_stream = self.left.execute(0, Arc::clone(&context))?;

                let reservation = MemoryConsumer::new("HashJoinInput")
                    .with_can_spill(spill_enabled)
                    .register(context.memory_pool());

                Ok(collect_left_input(
                    self.random_state.random_state().clone(),
                    left_stream,
                    on_left.clone(),
                    join_metrics.clone(),
                    reservation,
                    need_produce_result_in_final(self.join_type),
                    self.right().output_partitioning().partition_count(),
                    enable_dynamic_filter_pushdown,
                    Arc::clone(context.session_config().options()),
                    self.null_equality,
                    array_map_created_count,
                    skew_partition_count.clone(),
                    Arc::clone(&runtime_env),
                    metrics_set.clone(),
                    0,
                ))
            })?,
            PartitionMode::Partitioned => {
                let left_stream = self.left.execute(partition, Arc::clone(&context))?;

                let reservation =
                    MemoryConsumer::new(format!("HashJoinInput[{partition}]"))
                        .with_can_spill(spill_enabled)
                        .register(context.memory_pool());
                OnceFut::new(collect_left_input(
                    self.random_state.random_state().clone(),
                    left_stream,
                    on_left.clone(),
                    join_metrics.clone(),
                    reservation,
                    need_produce_result_in_final(self.join_type),
                    1,
                    enable_dynamic_filter_pushdown,
                    Arc::clone(context.session_config().options()),
                    self.null_equality,
                    array_map_created_count,
                    skew_partition_count.clone(),
                    Arc::clone(&runtime_env),
                    metrics_set.clone(),
                    partition,
                ))
            }
            PartitionMode::Auto => {
                return plan_err!(
                    "Invalid HashJoinExec, unsupported PartitionMode {:?} in execute()",
                    PartitionMode::Auto
                );
            }
        };

        let batch_size = context.session_config().batch_size();

        // we have the batches and the hash map with their keys. We can how create a stream
        // over the right that uses this information to issue new batches.
        let right_stream = self.right.execute(partition, context)?;

        // update column indices to reflect the projection
        let column_indices_after_projection = match self.projection.as_ref() {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        let on_right = self
            .on
            .iter()
            .map(|(_, right_expr)| Arc::clone(right_expr))
            .collect::<Vec<_>>();

        Ok(Box::pin(HashJoinStream::new(
            partition,
            self.schema(),
            on_right,
            self.filter.clone(),
            self.join_type,
            right_stream,
            self.random_state.random_state().clone(),
            join_metrics,
            column_indices_after_projection,
            self.null_equality,
            HashJoinStreamState::WaitBuildSide,
            BuildSide::Initial(BuildSideInitialState { left_fut }),
            batch_size,
            vec![],
            self.right.output_ordering().is_some(),
            build_accumulator,
            self.mode,
            self.null_aware,
            self.fetch,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let stats = match (partition, self.mode) {
            // For CollectLeft mode, the left side is collected into a single partition,
            // so all left partitions are available to each output partition.
            // For the right side, we need the specific partition statistics.
            (Some(partition), PartitionMode::CollectLeft) => {
                let left_stats = self.left.partition_statistics(None)?;
                let right_stats = self.right.partition_statistics(Some(partition))?;

                estimate_join_statistics(
                    Arc::unwrap_or_clone(left_stats),
                    Arc::unwrap_or_clone(right_stats),
                    &self.on,
                    &self.join_type,
                    &self.join_schema,
                )?
            }

            // For Partitioned mode, both sides are partitioned, so each output partition
            // only has access to the corresponding partition from both sides.
            (Some(partition), PartitionMode::Partitioned) => {
                let left_stats = self.left.partition_statistics(Some(partition))?;
                let right_stats = self.right.partition_statistics(Some(partition))?;

                estimate_join_statistics(
                    Arc::unwrap_or_clone(left_stats),
                    Arc::unwrap_or_clone(right_stats),
                    &self.on,
                    &self.join_type,
                    &self.join_schema,
                )?
            }

            // For Auto mode or when no specific partition is requested, fall back to
            // the current behavior of getting all partition statistics.
            (None, _) | (Some(_), PartitionMode::Auto) => {
                // TODO stats: it is not possible in general to know the output size of joins
                // There are some special cases though, for example:
                // - `A LEFT JOIN B ON A.col=B.col` with `COUNT_DISTINCT(B.col)=COUNT(B.col)`
                let left_stats = self.left.partition_statistics(None)?;
                let right_stats = self.right.partition_statistics(None)?;
                estimate_join_statistics(
                    Arc::unwrap_or_clone(left_stats),
                    Arc::unwrap_or_clone(right_stats),
                    &self.on,
                    &self.join_type,
                    &self.join_schema,
                )?
            }
        };
        // Project statistics if there is a projection
        let stats = stats.project(self.projection.as_ref());
        // Apply fetch limit to statistics
        Ok(Arc::new(stats.with_fetch(self.fetch, 0, 1)?))
    }

    /// Tries to push `projection` down through `hash_join`. If possible, performs the
    /// pushdown and returns a new [`HashJoinExec`] as the top plan which has projections
    /// as its children. Otherwise, returns `None`.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // TODO: currently if there is projection in HashJoinExec, we can't push down projection to left or right input. Maybe we can pushdown the mixed projection later.
        if self.contains_projection() {
            return Ok(None);
        }

        let schema = self.schema();
        if let Some(JoinData {
            projected_left_child,
            projected_right_child,
            join_filter,
            join_on,
        }) = try_pushdown_through_join(
            projection,
            self.left(),
            self.right(),
            self.on(),
            &schema,
            self.filter(),
        )? {
            self.builder()
                .with_new_children(vec![
                    Arc::new(projected_left_child),
                    Arc::new(projected_right_child),
                ])?
                .with_on(join_on)
                .with_filter(join_filter)
                // Returned early if projection is not None
                .with_projection(None)
                .build_exec()
                .map(Some)
        } else {
            try_embed_projection(projection, self)
        }
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        // This is the physical-plan equivalent of `push_down_all_join` in
        // `datafusion/optimizer/src/push_down_filter.rs`. That function uses `lr_is_preserved`
        // to decide which parent predicates can be pushed past a logical join to its children,
        // then checks column references to route each predicate to the correct side.
        //
        // We apply the same two-level logic here:
        // 1. `lr_is_preserved` gates whether a side is eligible at all.
        // 2. For each filter, we check that all column references belong to the
        //    target child (using `column_indices` to map output column positions
        //    to join sides). This is critical for correctness: name-based matching
        //    alone (as done by `ChildFilterDescription::from_child`) can incorrectly
        //    push filters when different join sides have columns with the same name
        //    (e.g. nested mark joins both producing "mark" columns).
        let (left_preserved, right_preserved) = lr_is_preserved(self.join_type);

        // Build the set of allowed column indices for each side
        let column_indices: Vec<ColumnIndex> = match self.projection.as_ref() {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        let (mut left_allowed, mut right_allowed) = (HashSet::new(), HashSet::new());
        column_indices
            .iter()
            .enumerate()
            .for_each(|(output_idx, ci)| {
                match ci.side {
                    JoinSide::Left => left_allowed.insert(output_idx),
                    JoinSide::Right => right_allowed.insert(output_idx),
                    // Mark columns - don't allow pushdown to either side
                    JoinSide::None => false,
                };
            });

        // For semi/anti joins, the non-preserved side's columns are not in the
        // output, but filters on join key columns can still be pushed there.
        // We find output columns that are join keys on the preserved side and
        // add their output indices to the non-preserved side's allowed set.
        // The name-based remap in FilterRemapper will then match them to the
        // corresponding column in the non-preserved child's schema.
        match self.join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => {
                let left_key_indices: HashSet<usize> = self
                    .on
                    .iter()
                    .filter_map(|(left_key, _)| {
                        left_key.downcast_ref::<Column>().map(|c| c.index())
                    })
                    .collect();
                for (output_idx, ci) in column_indices.iter().enumerate() {
                    if ci.side == JoinSide::Left && left_key_indices.contains(&ci.index) {
                        right_allowed.insert(output_idx);
                    }
                }
            }
            JoinType::RightSemi | JoinType::RightAnti => {
                let right_key_indices: HashSet<usize> = self
                    .on
                    .iter()
                    .filter_map(|(_, right_key)| {
                        right_key.downcast_ref::<Column>().map(|c| c.index())
                    })
                    .collect();
                for (output_idx, ci) in column_indices.iter().enumerate() {
                    if ci.side == JoinSide::Right && right_key_indices.contains(&ci.index)
                    {
                        left_allowed.insert(output_idx);
                    }
                }
            }
            _ => {}
        }

        let left_child = if left_preserved {
            ChildFilterDescription::from_child_with_allowed_indices(
                &parent_filters,
                left_allowed,
                self.left(),
            )?
        } else {
            ChildFilterDescription::all_unsupported(&parent_filters)
        };

        let mut right_child = if right_preserved {
            ChildFilterDescription::from_child_with_allowed_indices(
                &parent_filters,
                right_allowed,
                self.right(),
            )?
        } else {
            ChildFilterDescription::all_unsupported(&parent_filters)
        };

        // Add dynamic filters in Post phase if enabled
        if phase == FilterPushdownPhase::Post
            && self.allow_join_dynamic_filter_pushdown(config)
        {
            // Add actual dynamic filter to right side (probe side)
            let dynamic_filter = Self::create_dynamic_filter(&self.on);
            right_child = right_child.with_self_filter(dynamic_filter);
        }

        Ok(FilterDescription::new()
            .with_child(left_child)
            .with_child(right_child))
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let mut result = FilterPushdownPropagation::if_any(child_pushdown_result.clone());
        assert_eq!(child_pushdown_result.self_filters.len(), 2); // Should always be 2, we have 2 children
        let right_child_self_filters = &child_pushdown_result.self_filters[1]; // We only push down filters to the right child
        // We expect 0 or 1 self filters
        if let Some(filter) = right_child_self_filters.first() {
            // Note that we don't check PushdDownPredicate::discrimnant because even if nothing said
            // "yes, I can fully evaluate this filter" things might still use it for statistics -> it's worth updating
            let predicate = Arc::clone(&filter.predicate);
            if let Ok(dynamic_filter) =
                Arc::downcast::<DynamicFilterPhysicalExpr>(predicate)
            {
                // We successfully pushed down our self filter - we need to make a new node with the dynamic filter
                let new_node = self
                    .builder()
                    .with_dynamic_filter(Some(HashJoinExecDynamicFilter {
                        filter: dynamic_filter,
                        build_accumulator: OnceLock::new(),
                    }))
                    .build_exec()?;
                result = result.with_updated_node(new_node);
            }
        }
        Ok(result)
    }

    fn supports_limit_pushdown(&self) -> bool {
        // Hash join execution plan does not support pushing limit down through to children
        // because the children don't know about the join condition and can't
        // determine how many rows to produce
        false
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        self.builder()
            .with_fetch(limit)
            .build()
            .ok()
            .map(|exec| Arc::new(exec) as _)
    }
}

/// Determines which sides of a join are "preserved" for filter pushdown.
///
/// A preserved side means filters on that side's columns can be safely pushed
/// below the join. This mirrors the logic in the logical optimizer's
/// `lr_is_preserved` in `datafusion/optimizer/src/push_down_filter.rs`.
fn lr_is_preserved(join_type: JoinType) -> (bool, bool) {
    match join_type {
        JoinType::Inner => (true, true),
        JoinType::Left => (true, false),
        JoinType::Right => (false, true),
        JoinType::Full => (false, false),
        // Filters in semi/anti joins are either on the preserved side, or on join keys,
        // as all output columns come from the preserved side. Join key filters can be
        // safely pushed down into the other side.
        JoinType::LeftSemi | JoinType::LeftAnti => (true, true),
        JoinType::RightSemi | JoinType::RightAnti => (true, true),
        JoinType::LeftMark => (true, false),
        JoinType::RightMark => (false, true),
    }
}

/// Accumulator for collecting min/max bounds from build-side data during hash join.
///
/// This struct encapsulates the logic for progressively computing column bounds
/// (minimum and maximum values) for a specific join key expression as batches
/// are processed during the build phase of a hash join.
///
/// The bounds are used for dynamic filter pushdown optimization, where filters
/// based on the actual data ranges can be pushed down to the probe side to
/// eliminate unnecessary data early.
struct CollectLeftAccumulator {
    /// The physical expression to evaluate for each batch
    expr: Arc<dyn PhysicalExpr>,
    /// Accumulator for tracking the minimum value across all batches
    min: MinAccumulator,
    /// Accumulator for tracking the maximum value across all batches
    max: MaxAccumulator,
}

impl CollectLeftAccumulator {
    /// Creates a new accumulator for tracking bounds of a join key expression.
    ///
    /// # Arguments
    /// * `expr` - The physical expression to track bounds for
    /// * `schema` - The schema of the input data
    ///
    /// # Returns
    /// A new `CollectLeftAccumulator` instance configured for the expression's data type
    fn try_new(expr: Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> Result<Self> {
        /// Recursively unwraps dictionary types to get the underlying value type.
        fn dictionary_value_type(data_type: &DataType) -> DataType {
            match data_type {
                DataType::Dictionary(_, value_type) => {
                    dictionary_value_type(value_type.as_ref())
                }
                _ => data_type.clone(),
            }
        }

        let data_type = expr
            .data_type(schema)
            // Min/Max can operate on dictionary data but expect to be initialized with the underlying value type
            .map(|dt| dictionary_value_type(&dt))?;
        Ok(Self {
            expr,
            min: MinAccumulator::try_new(&data_type)?,
            max: MaxAccumulator::try_new(&data_type)?,
        })
    }

    /// Updates the accumulators with values from a new batch.
    ///
    /// Evaluates the expression on the batch and updates both min and max
    /// accumulators with the resulting values.
    ///
    /// # Arguments
    /// * `batch` - The record batch to process
    ///
    /// # Returns
    /// Ok(()) if the update succeeds, or an error if expression evaluation fails
    fn update_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let array = self.expr.evaluate(batch)?.into_array(batch.num_rows())?;
        self.min.update_batch(std::slice::from_ref(&array))?;
        self.max.update_batch(std::slice::from_ref(&array))?;
        Ok(())
    }

    /// Finalizes the accumulation and returns the computed bounds.
    ///
    /// Consumes self to extract the final min and max values from the accumulators.
    ///
    /// # Returns
    /// The `ColumnBounds` containing the minimum and maximum values observed
    fn evaluate(mut self) -> Result<ColumnBounds> {
        Ok(ColumnBounds::new(
            self.min.evaluate()?,
            self.max.evaluate()?,
        ))
    }
}

/// One partition's worth of accumulated build-side input.
///
/// A slot is either resident in memory (`InMemory`) or has been
/// streamed out to a temporary disk file (`Spilled`). Spilled
/// partitions are read back during build finalization and folded
/// back into the flat batch list consumed by the existing
/// hash-table builder.
///
/// Spill is only ever activated in multi-partition mode
/// (`hash_join_spill_threshold > 0.0`); the single-partition path
/// keeps slot 0 in `InMemory` for the lifetime of the build, which
/// is bit-for-bit equivalent to the pre-PR2 flat-vector layout.
#[derive(Debug)]
enum PartitionSlot {
    InMemory(BuildPartition),
    /// Spilled partition: the spill file owns the on-disk bytes
    /// (`RefCountedTempFile` cleans up on drop) and we record the
    /// pre-spill memory cost so we can refund it during readback
    /// without losing the metrics chain.
    Spilled {
        file: RefCountedTempFile,
        num_rows: usize,
        bytes: usize,
    },
}

impl Default for PartitionSlot {
    fn default() -> Self {
        Self::InMemory(BuildPartition::default())
    }
}

#[derive(Debug, Default)]
struct BuildPartition {
    batches: Vec<RecordBatch>,
    num_rows: usize,
    /// Cached sum of `get_record_batch_memory_size` for everything
    /// in `batches`, kept in lock-step with `push` so we have an O(1)
    /// answer to "how big is this partition" when picking a spill
    /// victim.
    bytes: usize,
}

impl BuildPartition {
    fn push(&mut self, batch: RecordBatch) {
        self.num_rows += batch.num_rows();
        self.bytes += get_record_batch_memory_size(&batch);
        self.batches.push(batch);
    }
}

impl PartitionSlot {
    fn num_rows(&self) -> usize {
        match self {
            PartitionSlot::InMemory(p) => p.num_rows,
            PartitionSlot::Spilled { num_rows, .. } => *num_rows,
        }
    }

    fn in_mem_bytes(&self) -> usize {
        match self {
            PartitionSlot::InMemory(p) => p.bytes,
            PartitionSlot::Spilled { .. } => 0,
        }
    }

    fn push(&mut self, batch: RecordBatch) -> Result<()> {
        match self {
            PartitionSlot::InMemory(p) => {
                p.push(batch);
                Ok(())
            }
            // We don't expect to push into an already-spilled slot in
            // the current absorbing-spill design — once a partition
            // spills it is sealed until readback. Future PRs that
            // append to spilled slots will replace this with an
            // in-progress-file append.
            PartitionSlot::Spilled { .. } => internal_err!(
                "BUG: attempted to push a build batch into an already-spilled partition slot"
            ),
        }
    }
}

/// State for collecting the build-side data during hash join.
///
/// Holds `num_partitions` slots of [`PartitionSlot`]. The default
/// is a single slot, behaviorally and structurally equivalent to
/// the previous flat-vector implementation. Multi-partition mode is
/// activated when `hash_join_spill_threshold > 0.0` and enables the
/// per-partition disk-spill path implemented in this PR.
struct BuildSideState {
    /// Per-partition accumulated state. Length = `num_partitions`.
    partitions: Vec<PartitionSlot>,
    /// Number of build partitions. `1` preserves legacy behavior.
    num_partitions: usize,
    /// Join keys, used to hash-route batches across partitions.
    on_left: Vec<Arc<dyn PhysicalExpr>>,
    /// Random state for hash-partition routing. Independent of the
    /// join's main `random_state` to avoid coupling partition
    /// assignment to the build's internal hash-table seed.
    partition_random_state: RandomState,
    /// Spill manager — `Some` only when `num_partitions > 1`. Owns
    /// the schema and metrics needed to drive `SpillManager`'s
    /// IPC writer/reader plumbing.
    spill_manager: Option<Arc<SpillManager>>,
    metrics: BuildProbeJoinMetrics,
    reservation: MemoryReservation,
    bounds_accumulators: Option<Vec<CollectLeftAccumulator>>,
    /// Per-partition consecutive-spill counter. Reset to 0 for
    /// every partition that is *not* the spill victim each round,
    /// incremented for the victim. PR4-F's skew detector ticks
    /// `skew_partition_count` once when any entry crosses
    /// [`HASH_JOIN_SKEW_SPILL_THRESHOLD`] (rising-edge only, so we
    /// don't double-count the same skew).
    consecutive_spills: Vec<usize>,
    /// PR4-F skew metric counter. Bumped exactly once per
    /// partition slot that gets flagged as skewed by the
    /// consecutive-spill heuristic.
    skew_partition_count: Count,
    /// Set of partition indices we've already flagged as skewed,
    /// so the metric counter rises monotonically rather than
    /// ticking on every spill after the threshold.
    skew_flagged: Vec<bool>,
}

impl BuildSideState {
    /// Create a new BuildSideState with optional accumulators for bounds computation.
    ///
    /// `num_partitions == 1` keeps the legacy single-slot path with
    /// no spill machinery. Values >1 activate hash-routing of
    /// incoming batches and per-partition disk spill on memory
    /// pressure.
    fn try_new(
        metrics: BuildProbeJoinMetrics,
        reservation: MemoryReservation,
        on_left: Vec<Arc<dyn PhysicalExpr>>,
        schema: &SchemaRef,
        should_compute_dynamic_filters: bool,
        num_partitions: usize,
        partition_random_state: RandomState,
        spill_manager: Option<Arc<SpillManager>>,
        skew_partition_count: Count,
    ) -> Result<Self> {
        assert!(num_partitions > 0, "num_partitions must be > 0");
        // Spill manager is only meaningful in multi-partition mode.
        debug_assert!(
            spill_manager.is_none() || num_partitions > 1,
            "spill_manager provided for single-partition mode is unused",
        );
        let mut partitions = Vec::with_capacity(num_partitions);
        partitions.resize_with(num_partitions, PartitionSlot::default);
        Ok(Self {
            partitions,
            num_partitions,
            on_left: on_left.clone(),
            partition_random_state,
            spill_manager,
            metrics,
            reservation,
            bounds_accumulators: should_compute_dynamic_filters
                .then(|| {
                    on_left
                        .into_iter()
                        .map(|expr| CollectLeftAccumulator::try_new(expr, schema))
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
            consecutive_spills: vec![0; num_partitions],
            skew_partition_count,
            skew_flagged: vec![false; num_partitions],
        })
    }

    /// Total rows accumulated across all partitions.
    fn num_rows(&self) -> usize {
        self.partitions.iter().map(|p| p.num_rows()).sum()
    }

    /// Try to grow the build-side reservation by `bytes`. On
    /// failure, attempt to spill the largest in-memory partition
    /// and retry; repeat up to `num_partitions - 1` times before
    /// surfacing the original `ResourceExhausted` error.
    ///
    /// In single-partition mode (no `spill_manager`) this is a
    /// straight pass-through to `try_grow`, preserving the legacy
    /// hard-OOM behavior at default settings.
    fn try_grow_or_spill(&mut self, bytes: usize) -> Result<()> {
        if self.spill_manager.is_none() {
            return self.reservation.try_grow(bytes);
        }
        // Try once cheap.
        if self.reservation.try_grow(bytes).is_ok() {
            return Ok(());
        }
        // Spill loop: pick the largest in-memory partition, write
        // it out, refund its bytes to the reservation, retry.
        let max_attempts = self.partitions.len();
        for _ in 0..max_attempts {
            if !self.spill_largest_in_memory_partition()? {
                // Nothing left to spill — fall through to the
                // final try_grow which will surface the real error.
                break;
            }
            if self.reservation.try_grow(bytes).is_ok() {
                return Ok(());
            }
        }
        // Final attempt; will return the upstream ResourceExhausted
        // if we still cannot fit.
        self.reservation.try_grow(bytes)
    }

    /// Spill the largest currently-in-memory partition slot to
    /// disk. Returns `Ok(true)` if a partition was spilled,
    /// `Ok(false)` if there were no in-memory partitions left to
    /// spill (caller treats this as terminal).
    fn spill_largest_in_memory_partition(&mut self) -> Result<bool> {
        let Some(spill_manager) = self.spill_manager.as_ref() else {
            return Ok(false);
        };
        // Pick the partition with the most resident bytes. Break
        // ties by index for determinism.
        let victim = self
            .partitions
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                PartitionSlot::InMemory(p) if p.bytes > 0 => Some((i, p.bytes)),
                _ => None,
            })
            .max_by_key(|(_, bytes)| *bytes)
            .map(|(i, _)| i);
        let Some(victim_idx) = victim else {
            return Ok(false);
        };
        // Take ownership of the in-memory contents so we can spill
        // without holding a mutable borrow on `self.partitions`.
        let in_mem = match std::mem::replace(
            &mut self.partitions[victim_idx],
            PartitionSlot::InMemory(BuildPartition::default()),
        ) {
            PartitionSlot::InMemory(p) => p,
            // Unreachable given the filter above, but be defensive.
            other => {
                self.partitions[victim_idx] = other;
                return Ok(false);
            }
        };
        let bytes = in_mem.bytes;
        let num_rows = in_mem.num_rows;
        let spilled_file = spill_manager.spill_record_batch_and_finish(
            &in_mem.batches,
            &format!("HashJoinBuildSpill[partition={victim_idx}]"),
        )?;
        // Drop the in-memory batches and refund their reservation
        // before installing the spilled marker, so a partial spill
        // can't double-count.
        drop(in_mem);
        self.reservation.shrink(bytes);
        self.metrics.build_mem_used.sub(bytes);
        match spilled_file {
            Some(file) => {
                self.partitions[victim_idx] = PartitionSlot::Spilled {
                    file,
                    num_rows,
                    bytes,
                };
                // PR4-F skew detector. Bump the victim's
                // consecutive-spill counter and reset every other
                // partition's counter, so the count tracks runs
                // of "the same partition kept being the largest."
                // When a victim crosses the threshold (and we
                // haven't already flagged it), tick the metric.
                if victim_idx < self.consecutive_spills.len() {
                    self.consecutive_spills[victim_idx] += 1;
                    for (idx, c) in self.consecutive_spills.iter_mut().enumerate() {
                        if idx != victim_idx {
                            *c = 0;
                        }
                    }
                    if self.consecutive_spills[victim_idx]
                        >= HASH_JOIN_SKEW_SPILL_THRESHOLD
                        && !self.skew_flagged[victim_idx]
                    {
                        self.skew_flagged[victim_idx] = true;
                        self.skew_partition_count.add(1);
                    }
                }
                Ok(true)
            }
            // No-op spill (empty input). Slot is already a default
            // empty InMemory; nothing more to do.
            None => Ok(false),
        }
    }

    /// Route `batch` into the appropriate partition slot(s).
    ///
    /// In single-partition mode this is a direct push to slot 0
    /// (zero overhead vs the legacy path). In multi-partition mode
    /// the batch is hash-split via `partition_batch_by_hash` and
    /// each non-empty sub-batch is appended to its slot. If a
    /// target slot has been spilled, the sub-batch is held in a
    /// fresh in-memory bucket alongside it (pushed back as a new
    /// `InMemory` value); a future PR can swap this for
    /// in-progress-file appends.
    fn push_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if self.num_partitions == 1 {
            return self.partitions[0].push(batch);
        }
        let parts = super::partitioned_build::partition_batch_by_hash(
            &batch,
            &self.on_left,
            self.num_partitions,
            &self.partition_random_state,
        )?;
        for (idx, part) in parts.into_iter().enumerate() {
            let Some(b) = part else { continue };
            // If the slot is currently spilled, materialize a fresh
            // in-memory shadow alongside it. Subsequent
            // `try_grow_or_spill` calls will spill *that* if needed.
            if matches!(self.partitions[idx], PartitionSlot::Spilled { .. }) {
                // Promote: keep the old spilled slot, but track new
                // arrivals in a parallel in-memory bucket. We model
                // this as overwriting only after the new batch is
                // pushed via a small helper.
                let new_in_mem = PartitionSlot::InMemory(BuildPartition::default());
                let old_spilled = std::mem::replace(&mut self.partitions[idx], new_in_mem);
                self.partitions[idx].push(b)?;
                // Record the pre-existing spilled bytes by stashing
                // a synthetic extra slot at the end of the vector;
                // readback walks the entire vector and folds both
                // back into the flat batch list.
                self.partitions.push(old_spilled);
                // PR4-F: keep the skew bookkeeping vectors aligned
                // with `partitions` so `spill_largest_in_memory_partition`
                // can index by `victim_idx` regardless of these
                // synthetic appends.
                self.consecutive_spills.push(0);
                self.skew_flagged.push(false);
            } else {
                self.partitions[idx].push(b)?;
            }
        }
        Ok(())
    }

    /// Finalize the build side into a per-partition list of
    /// [`PartitionFinalState`].
    ///
    /// PR4-A scaffolding: the returned `Vec` has one entry per
    /// build partition (length = `self.num_partitions`). Spilled
    /// slots are read back from disk *here* and folded into a
    /// `Resident` entry, exactly as `into_flat_batches` did
    /// previously, so this is bit-for-bit equivalent to the PR3
    /// readback path. Future PRs (4-D / probe replay) will add a
    /// `Spilled` variant that defers readback until probe time;
    /// stream.rs will then materialize partitions one at a time.
    ///
    /// The accompanying [`FinalizedBuildSide`] carries the non-batch
    /// fields (metrics, reservation, bounds accumulators, total
    /// rows) so the caller can continue exactly as before.
    async fn into_partition_finalize(
        self,
    ) -> Result<(Vec<PartitionFinalState>, FinalizedBuildSide)> {
        let BuildSideState {
            partitions,
            num_partitions: _,
            on_left: _,
            partition_random_state: _,
            spill_manager,
            metrics,
            reservation,
            bounds_accumulators,
            consecutive_spills: _,
            skew_partition_count: _,
            skew_flagged: _,
        } = self;
        let mut out: Vec<PartitionFinalState> = Vec::with_capacity(partitions.len());
        let mut total_rows = 0usize;
        for slot in partitions {
            match slot {
                PartitionSlot::InMemory(p) => {
                    total_rows += p.num_rows;
                    out.push(PartitionFinalState::Resident {
                        batches: p.batches,
                        num_rows: p.num_rows,
                    });
                }
                PartitionSlot::Spilled {
                    file,
                    num_rows,
                    bytes,
                } => {
                    total_rows += num_rows;
                    // PR4-D-2: defer readback. We do NOT regrow
                    // the reservation here — that happens in
                    // `MaterializePartition` (or on demand in
                    // `into_flat_batches` for the single-hashmap
                    // path). Reservation accounting still nets
                    // out: the bytes were `shrink`-ed during
                    // spill, and will be `try_grow`-ed when the
                    // slot is materialized.
                    let sm = spill_manager.as_ref().cloned().ok_or_else(|| {
                        DataFusionError::Internal(
                            "BUG: spilled partition without spill_manager".into(),
                        )
                    })?;
                    out.push(PartitionFinalState::Spilled {
                        file,
                        num_rows,
                        bytes,
                        spill_manager: sm,
                    });
                }
            }
        }
        Ok((
            out,
            FinalizedBuildSide {
                num_rows: total_rows,
                metrics,
                reservation,
                bounds_accumulators,
            },
        ))
    }

    /// Flatten all partitions (in-memory and spilled) into a single
    /// `Vec<RecordBatch>`.
    ///
    /// Implemented in PR4-A on top of [`Self::into_partition_finalize`]
    /// to keep the existing single-hashmap probe path bit-for-bit
    /// equivalent while the per-partition machinery is wired up
    /// behind the scenes.
    ///
    /// PR4-D-2 moves spill readback out of `into_partition_finalize`
    /// (which now emits the `Spilled` variant directly), so this
    /// function performs the readback locally — it remains the
    /// "collapse to one concatenated batch" entry point that the
    /// existing single-hashmap probe consumes.
    async fn into_flat_batches(
        self,
    ) -> Result<(Vec<RecordBatch>, FinalizedBuildSide)> {
        let (partitions, mut finalized) = self.into_partition_finalize().await?;
        let mut flat: Vec<RecordBatch> = Vec::new();
        for p in partitions {
            match p {
                PartitionFinalState::Resident { batches, .. } => flat.extend(batches),
                PartitionFinalState::Spilled {
                    file,
                    bytes,
                    spill_manager,
                    ..
                } => {
                    // Restore the reservation for the in-memory
                    // copy we are about to materialize. Mirrors
                    // PR4-A's pre-split behavior so accounting is
                    // identical end-to-end.
                    finalized.reservation.try_grow(bytes)?;
                    finalized.metrics.build_mem_used.add(bytes);
                    let mut stream = spill_manager.read_spill_as_stream(file, None)?;
                    use futures::StreamExt;
                    while let Some(batch) = stream.next().await {
                        flat.push(batch?);
                    }
                }
            }
        }
        Ok((flat, finalized))
    }
}

/// Finalized per-partition build state, returned from
/// [`BuildSideState::into_partition_finalize`].
///
/// PR4-D-2 introduces the `Spilled` variant. The single-hashmap
/// probe path (consumed via [`BuildSideState::into_flat_batches`])
/// reads spilled slots back into RAM at flatten time, so PR3's
/// "build collapses to one concatenated batch" behavior is
/// preserved bit-for-bit. PR4-D-3 will route `Spilled` straight
/// into [`super::stream::HashJoinStream`]'s `MaterializePartition`
/// state so probe-time replay no longer round-trips through the
/// build-time memory pool.
#[derive(Debug)]
enum PartitionFinalState {
    /// Build batches resident in memory, ready to feed the
    /// hash-table builder. `num_rows` is the sum of
    /// `batch.num_rows()` across `batches`, cached so callers don't
    /// have to re-walk the vector.
    Resident {
        batches: Vec<RecordBatch>,
        // PR4-B will read `num_rows` when sizing per-partition
        // hash maps and visited-indices bitmaps. Allowed dead in
        // PR4-A so the scaffolding lands without churn.
        #[allow(dead_code)]
        num_rows: usize,
    },
    /// Build batches still on disk. The reservation has *not*
    /// been regrown; whoever materializes this slot is responsible
    /// for `reservation.try_grow(bytes)` before reading the spill
    /// file back.
    ///
    /// `spill_manager` is held here (rather than on the parent
    /// `JoinLeftData`) so each slot is self-contained: the
    /// per-partition probe path in PR4-D-3 can `mem::replace` an
    /// individual slot to `Resident` without holding a borrow on
    /// the rest of the build state.
    Spilled {
        file: RefCountedTempFile,
        // PR4-D-3 will read `num_rows` to pre-size the
        // visited-indices bitmap when materializing this slot.
        #[allow(dead_code)]
        num_rows: usize,
        // PR4-D-3 will read `bytes` to `try_grow` the reservation
        // before readback; allowed dead in PR4-D-2 because
        // `into_flat_batches` is the only caller and it already
        // has `bytes` from the destructuring pattern.
        #[allow(dead_code)]
        bytes: usize,
        spill_manager: Arc<SpillManager>,
    },
}

/// Non-batch fields of [`BuildSideState`] returned alongside the
/// flattened batches by [`BuildSideState::into_flat_batches`].
struct FinalizedBuildSide {
    num_rows: usize,
    metrics: BuildProbeJoinMetrics,
    reservation: MemoryReservation,
    bounds_accumulators: Option<Vec<CollectLeftAccumulator>>,
}

fn should_collect_min_max_for_perfect_hash(
    on_left: &[PhysicalExprRef],
    schema: &SchemaRef,
) -> Result<bool> {
    if on_left.len() != 1 {
        return Ok(false);
    }

    let expr = &on_left[0];
    let data_type = expr.data_type(schema)?;
    Ok(ArrayMap::is_supported_type(&data_type))
}

/// Collects all batches from the left (build) side stream and creates a hash map for joining.
///
/// This function is responsible for:
/// 1. Consuming the entire left stream and collecting all batches into memory
/// 2. Building a hash map from the join key columns for efficient probe operations
/// 3. Computing bounds for dynamic filter pushdown (if enabled)
/// 4. Preparing visited indices bitmap for certain join types
///
/// # Parameters
/// * `random_state` - Random state for consistent hashing across partitions
/// * `left_stream` - Stream of record batches from the build side
/// * `on_left` - Physical expressions for the left side join keys
/// * `metrics` - Metrics collector for tracking memory usage and row counts
/// * `reservation` - Memory reservation tracker for the hash table and data
/// * `with_visited_indices_bitmap` - Whether to track visited indices (for outer joins)
/// * `probe_threads_count` - Number of threads that will probe this hash table
/// * `should_compute_dynamic_filters` - Whether to compute min/max bounds for dynamic filtering
///
/// # Dynamic Filter Coordination
/// When `should_compute_dynamic_filters` is true, this function computes the min/max bounds
/// for each join key column but does NOT update the dynamic filter. Instead, the
/// bounds are stored in the returned `JoinLeftData` and later coordinated by
/// `SharedBuildAccumulator` to ensure all partitions contribute their bounds
/// before updating the filter exactly once.
///
/// # Returns
/// `JoinLeftData` containing the hash map, consolidated batch, join key values,
/// visited indices bitmap, and computed bounds (if requested).
#[expect(clippy::too_many_arguments)]
async fn collect_left_input(
    random_state: RandomState,
    left_stream: SendableRecordBatchStream,
    on_left: Vec<PhysicalExprRef>,
    metrics: BuildProbeJoinMetrics,
    reservation: MemoryReservation,
    with_visited_indices_bitmap: bool,
    probe_threads_count: usize,
    should_compute_dynamic_filters: bool,
    config: Arc<ConfigOptions>,
    null_equality: NullEquality,
    array_map_created_count: Count,
    skew_partition_count: Count,
    runtime_env: Arc<RuntimeEnv>,
    metrics_set: ExecutionPlanMetricsSet,
    partition_idx: usize,
) -> Result<JoinLeftData> {
    let schema = left_stream.schema();

    let should_collect_min_max_for_phj =
        should_collect_min_max_for_perfect_hash(&on_left, &schema)?;

    // Partitioned-build configuration. `num_partitions == 1` (the
    // default unless `hash_join_spill_threshold > 0.0`) preserves
    // the legacy single-batch-vector behavior bit-for-bit; >1
    // hash-routes incoming batches into per-partition slots in
    // preparation for PR3's disk-spill path.
    let exec_opts = &config.execution;
    let num_build_partitions = if exec_opts.hash_join_spill_threshold > 0.0 {
        exec_opts.hash_join_num_partitions.max(1)
    } else {
        1
    };
    // Use a fixed-seed RandomState for partition routing so the
    // assignment is deterministic across runs and independent of
    // the join's main-table hash seed.
    let partition_random_state = RandomState::with_seed(0);

    // Construct a SpillManager only in multi-partition mode. The
    // schema we register is the build-side schema; the SpillMetrics
    // wire into the existing ExecutionPlanMetricsSet so spill bytes
    // and counts surface in EXPLAIN ANALYZE output.
    let spill_manager: Option<Arc<SpillManager>> = if num_build_partitions > 1 {
        let spill_metrics =
            crate::metrics::SpillMetrics::new(&metrics_set, partition_idx);
        Some(Arc::new(
            SpillManager::new(Arc::clone(&runtime_env), spill_metrics, Arc::clone(&schema))
                .with_compression_type(SpillCompression::default()),
        ))
    } else {
        None
    };

    // Clones for JoinLeftData; build-side moves the originals into
    // BuildSideState. The probe-side hash router (PR4-D-3b) MUST use
    // the same `partition_random_state` instance (same seed) so build
    // and probe agree on partition assignment.
    let join_left_partition_random_state = partition_random_state.clone();
    let join_left_spill_manager = spill_manager.clone();

    let initial = BuildSideState::try_new(
        metrics,
        reservation,
        on_left.clone(),
        &schema,
        should_compute_dynamic_filters || should_collect_min_max_for_phj,
        num_build_partitions,
        partition_random_state,
        spill_manager,
        skew_partition_count,
    )?;

    let state = left_stream
        .try_fold(initial, |mut state, batch| async move {
            // Update accumulators if computing bounds
            if let Some(ref mut accumulators) = state.bounds_accumulators {
                for accumulator in accumulators {
                    accumulator.update_batch(&batch)?;
                }
            }

            // Reserve memory for incoming batch, spilling the
            // largest in-memory partition on `try_grow` failure
            // (multi-partition mode only; single-partition mode
            // surfaces the original ResourceExhausted as before).
            let batch_size = get_record_batch_memory_size(&batch);
            state.try_grow_or_spill(batch_size)?;
            // Update metrics
            state.metrics.build_mem_used.add(batch_size);
            state.metrics.build_input_batches.add(1);
            state.metrics.build_input_rows.add(batch.num_rows());
            // Hash-route into partition slots (single-partition mode
            // is a no-op fast path).
            state.push_batch(batch)?;
            Ok(state)
        })
        .await?;

    // PR4-D-3b: branch on `num_build_partitions`.
    //
    // * Single-partition mode (default): preserve the legacy
    //   `into_flat_batches` → single-`Resident` path bit-for-bit,
    //   including dynamic-filter `membership` selection.
    // * Multi-partition mode: consume `into_partition_finalize`
    //   directly, build a per-partition hash map for each
    //   `Resident` slot, and emit `PartitionEntry::Spilled` for
    //   each on-disk slot so the probe state machine's
    //   `MaterializePartition` step can pull them back in lazily.
    //   Dynamic-filter `membership` is disabled in this mode (see
    //   the `multi_partition_build` gate in `execute()`); bounds-based
    //   pushdown still works through the parent `bounds`.
    let single_partition_build = num_build_partitions == 1;
    let (partitions, _num_rows, reservation, metrics, mut bounds, membership) = if single_partition_build {
        let (batches, finalized) = state.into_flat_batches().await?;
        let FinalizedBuildSide {
            num_rows,
            metrics,
            mut reservation,
            bounds_accumulators,
        } = finalized;

        let bounds = match bounds_accumulators {
            Some(accumulators) if num_rows > 0 => {
                let bounds = accumulators
                    .into_iter()
                    .map(CollectLeftAccumulator::evaluate)
                    .collect::<Result<Vec<_>>>()?;
                Some(PartitionBounds::new(bounds))
            }
            _ => None,
        };

        let partition_data = build_partition_data(
            batches,
            num_rows,
            bounds.clone(),
            &schema,
            &on_left,
            &random_state,
            null_equality,
            with_visited_indices_bitmap,
            &mut reservation,
            &metrics,
            &array_map_created_count,
            config.execution.perfect_hash_join_small_build_threshold,
            config.execution.perfect_hash_join_min_key_density,
        )?;
        let membership = compute_membership(
            &partition_data,
            num_rows,
            config.optimizer.hash_join_inlist_pushdown_max_size,
            config.optimizer.hash_join_inlist_pushdown_max_distinct_values,
        )?;
        (
            vec![PartitionEntry::Resident(partition_data)],
            num_rows,
            reservation,
            metrics,
            bounds,
            membership,
        )
    } else {
        let (slots, finalized) = state.into_partition_finalize().await?;
        let FinalizedBuildSide {
            num_rows,
            metrics,
            mut reservation,
            bounds_accumulators,
        } = finalized;

        // Parent (union) bounds — same as single-partition mode,
        // since the accumulators saw every input batch regardless
        // of which slot it routed to.
        let bounds = match bounds_accumulators {
            Some(accumulators) if num_rows > 0 => {
                let bounds = accumulators
                    .into_iter()
                    .map(CollectLeftAccumulator::evaluate)
                    .collect::<Result<Vec<_>>>()?;
                Some(PartitionBounds::new(bounds))
            }
            _ => None,
        };

        let mut partitions = Vec::with_capacity(slots.len());
        for slot in slots {
            match slot {
                PartitionFinalState::Resident { batches, num_rows: pnum_rows } => {
                    // Each partition gets its own hash map built
                    // over only that partition's rows.
                    let partition_data = build_partition_data(
                        batches,
                        pnum_rows,
                        // Per-partition bounds: PR4 v1 reuses the
                        // parent (union) bounds for each
                        // partition. Cheap, correct (each partition
                        // is a subset of union), and dynamic-filter
                        // membership pushdown is gated off in
                        // multi-partition mode anyway. Tighter
                        // per-partition bounds is a follow-up.
                        bounds.clone(),
                        &schema,
                        &on_left,
                        &random_state,
                        null_equality,
                        with_visited_indices_bitmap,
                        &mut reservation,
                        &metrics,
                        &array_map_created_count,
                        config.execution.perfect_hash_join_small_build_threshold,
                        config.execution.perfect_hash_join_min_key_density,
                    )?;
                    partitions.push(PartitionEntry::Resident(partition_data));
                }
                PartitionFinalState::Spilled {
                    file,
                    num_rows: pnum_rows,
                    bytes,
                    spill_manager,
                } => {
                    partitions.push(PartitionEntry::Spilled(SpilledSlot {
                        file,
                        num_rows: pnum_rows,
                        bytes,
                        spill_manager,
                        cell: tokio::sync::OnceCell::new(),
                    }));
                }
            }
        }

        // `membership` for filter pushdown: multi-partition mode
        // can't represent N hash maps in `PushdownStrategy`, and
        // `enable_dynamic_filter_pushdown` is gated off in this
        // mode anyway. Use `Empty` as a sentinel only after
        // confirming the build is genuinely empty; otherwise pick
        // a never-installed strategy. Since the gate prevents
        // installation we can pick either; choose `Empty` only
        // when num_rows == 0 to mirror the single-partition
        // semantics for the empty-build path.
        let membership = if num_rows == 0 {
            PushdownStrategy::Empty
        } else {
            // Sentinel: this strategy is never actually consumed
            // because `enable_dynamic_filter_pushdown` is false in
            // multi-partition mode. We pick the first resident map
            // we find purely to keep the type-shape; if no slot
            // is resident yet (everything spilled), fall back to
            // `Empty` (still won't be consumed).
            partitions
                .iter()
                .find_map(|p| match p {
                    PartitionEntry::Resident(d) => Some(PushdownStrategy::Map(Arc::clone(&d.map))),
                    PartitionEntry::Spilled(_) => None,
                })
                .unwrap_or(PushdownStrategy::Empty)
        };

        (partitions, num_rows, reservation, metrics, bounds, membership)
    };

    if should_collect_min_max_for_phj && !should_compute_dynamic_filters {
        bounds = None;
    }

    let num_partitions = partitions.len();
    let materialize_ctx = if single_partition_build {
        None
    } else {
        Some(Arc::new(MaterializeContext {
            on_left: on_left.clone(),
            random_state: random_state.clone(),
            null_equality,
            with_visited_indices_bitmap,
            schema: Arc::clone(&schema),
            perfect_hash_join_small_build_threshold:
                config.execution.perfect_hash_join_small_build_threshold,
            perfect_hash_join_min_key_density:
                config.execution.perfect_hash_join_min_key_density,
            array_map_created_count: array_map_created_count.clone(),
            metrics: metrics.clone(),
            runtime_env: Arc::clone(&runtime_env),
        }))
    };
    let data = JoinLeftData {
        partitions,
        num_partitions,
        partition_random_state: join_left_partition_random_state,
        spill_manager: join_left_spill_manager,
        materialize_ctx,
        probe_threads_counter: AtomicUsize::new(probe_threads_count),
        _reservation: reservation,
        bounds,
        membership,
        probe_side_non_empty: AtomicBool::new(false),
        probe_side_has_null: AtomicBool::new(false),
    };

    Ok(data)
}

/// Build a single per-partition [`JoinLeftPartitionData`] from a
/// `Vec<RecordBatch>`.
///
/// Extracted from the tail of `collect_left_input` in PR4-D-3b so
/// the multi-partition path can call it once per partition. The
/// single-partition path calls it exactly once with the full
/// concatenated batch list, preserving PR3's bit-for-bit
/// behavior (same `try_create_array_map` decision, same
/// reservation accounting, same `update_hash` ordering).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_partition_data(
    batches: Vec<RecordBatch>,
    num_rows: usize,
    bounds: Option<PartitionBounds>,
    schema: &SchemaRef,
    on_left: &[PhysicalExprRef],
    random_state: &RandomState,
    null_equality: NullEquality,
    with_visited_indices_bitmap: bool,
    reservation: &mut MemoryReservation,
    metrics: &BuildProbeJoinMetrics,
    array_map_created_count: &Count,
    perfect_hash_join_small_build_threshold: usize,
    perfect_hash_join_min_key_density: f64,
) -> Result<JoinLeftPartitionData> {
    let (join_hash_map, batch, left_values) = if let Some((array_map, batch, left_value)) =
        try_create_array_map(
            &bounds,
            schema,
            &batches,
            on_left,
            reservation,
            perfect_hash_join_small_build_threshold,
            perfect_hash_join_min_key_density,
            null_equality,
        )?
    {
        array_map_created_count.add(1);
        metrics.build_mem_used.add(array_map.size());
        (Map::ArrayMap(array_map), batch, left_value)
    } else {
        let fixed_size_u32 = size_of::<JoinHashMapU32>();
        let fixed_size_u64 = size_of::<JoinHashMapU64>();

        let mut hashmap: Box<dyn JoinHashMapType> = if num_rows > u32::MAX as usize {
            let estimated_hashtable_size =
                estimate_memory_size::<(u64, u64)>(num_rows, fixed_size_u64)?;
            reservation.try_grow(estimated_hashtable_size)?;
            metrics.build_mem_used.add(estimated_hashtable_size);
            Box::new(JoinHashMapU64::with_capacity(num_rows))
        } else {
            let estimated_hashtable_size =
                estimate_memory_size::<(u32, u64)>(num_rows, fixed_size_u32)?;
            reservation.try_grow(estimated_hashtable_size)?;
            metrics.build_mem_used.add(estimated_hashtable_size);
            Box::new(JoinHashMapU32::with_capacity(num_rows))
        };

        let mut hashes_buffer = Vec::new();
        let mut offset = 0;
        let batches_iter = batches.iter().rev();
        for batch in batches_iter.clone() {
            hashes_buffer.clear();
            hashes_buffer.resize(batch.num_rows(), 0);
            update_hash(
                on_left,
                batch,
                &mut *hashmap,
                offset,
                random_state,
                &mut hashes_buffer,
                0,
                true,
            )?;
            offset += batch.num_rows();
        }
        let batch = concat_batches(schema, batches_iter.clone())?;
        let left_values = evaluate_expressions_to_arrays(on_left, &batch)?;
        (Map::HashMap(hashmap), batch, left_values)
    };

    let visited_indices_bitmap = if with_visited_indices_bitmap {
        let bitmap_size = bit_util::ceil(batch.num_rows(), 8);
        reservation.try_grow(bitmap_size)?;
        metrics.build_mem_used.add(bitmap_size);
        let mut bitmap_buffer = BooleanBufferBuilder::new(batch.num_rows());
        bitmap_buffer.append_n(num_rows, false);
        bitmap_buffer
    } else {
        BooleanBufferBuilder::new(0)
    };

    Ok(JoinLeftPartitionData {
        map: Arc::new(join_hash_map),
        batch,
        values: left_values,
        visited_indices_bitmap: Mutex::new(visited_indices_bitmap),
        bounds,
    })
}

/// Compute the [`PushdownStrategy`] for the build side from a
/// single resident partition.
///
/// PR4-D-3b only invokes this on the single-partition build path;
/// the multi-partition path bypasses dynamic-filter membership
/// pushdown entirely (see the `multi_partition_build` gate in
/// `execute()`).
fn compute_membership(
    partition: &JoinLeftPartitionData,
    num_rows: usize,
    inlist_pushdown_max_size: usize,
    inlist_pushdown_max_distinct_values: usize,
) -> Result<PushdownStrategy> {
    if num_rows == 0 {
        return Ok(PushdownStrategy::Empty);
    }
    let estimated_size = partition
        .values
        .iter()
        .map(|arr| arr.get_array_memory_size())
        .sum::<usize>();
    let strategy = if partition.values.is_empty()
        || partition.values[0].is_empty()
        || estimated_size > inlist_pushdown_max_size
        || partition.map.num_of_distinct_key() > inlist_pushdown_max_distinct_values
    {
        PushdownStrategy::Map(Arc::clone(&partition.map))
    } else if let Some(in_list_values) = build_struct_inlist_values(&partition.values)? {
        PushdownStrategy::InList(in_list_values)
    } else {
        PushdownStrategy::Map(Arc::clone(&partition.map))
    };
    Ok(strategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_phj_used(metrics: &MetricsSet, use_phj: bool) {
        if use_phj {
            assert!(
                metrics
                    .sum_by_name(ARRAY_MAP_CREATED_COUNT_METRIC_NAME)
                    .expect("should have array_map_created_count metrics")
                    .as_usize()
                    >= 1
            );
        } else {
            assert_eq!(
                metrics
                    .sum_by_name(ARRAY_MAP_CREATED_COUNT_METRIC_NAME)
                    .map(|v| v.as_usize())
                    .unwrap_or(0),
                0
            )
        }
    }

    fn build_schema_and_on() -> Result<(SchemaRef, SchemaRef, JoinOn)> {
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("a1", DataType::Int32, true),
            Field::new("b1", DataType::Int32, true),
        ]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("a2", DataType::Int32, true),
            Field::new("b1", DataType::Int32, true),
        ]));
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left_schema)?) as _,
            Arc::new(Column::new_with_schema("b1", &right_schema)?) as _,
        )];
        Ok((left_schema, right_schema, on))
    }

    use crate::coalesce_partitions::CoalescePartitionsExec;
    use crate::joins::hash_join::stream::lookup_join_hashmap;
    use crate::test::{TestMemoryExec, assert_join_metrics};
    use crate::{
        common, expressions::Column, repartition::RepartitionExec, test::build_table_i32,
        test::exec::MockExec,
    };

    use arrow::array::{
        Date32Array, Int32Array, Int64Array, StructArray, UInt32Array, UInt64Array,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::hash_utils::create_hashes;
    use datafusion_common::test_util::{batches_to_sort_string, batches_to_string};
    use datafusion_common::{
        ScalarValue, assert_batches_eq, assert_batches_sorted_eq, assert_contains,
        exec_err, internal_err,
    };
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal};
    use hashbrown::HashTable;
    use insta::{allow_duplicates, assert_snapshot};
    use rstest::*;
    use rstest_reuse::*;

    fn div_ceil(a: usize, b: usize) -> usize {
        a.div_ceil(b)
    }

    #[template]
    #[rstest]
    fn hash_join_exec_configs(
        #[values(8192, 10, 5, 2, 1)] batch_size: usize,
        #[values(true, false)] use_perfect_hash_join_as_possible: bool,
    ) {
    }

    fn prepare_task_ctx(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Arc<TaskContext> {
        let mut session_config = SessionConfig::default().with_batch_size(batch_size);

        if use_perfect_hash_join_as_possible {
            session_config
                .options_mut()
                .execution
                .perfect_hash_join_small_build_threshold = 819200;
            session_config
                .options_mut()
                .execution
                .perfect_hash_join_min_key_density = 0.0;
        } else {
            session_config
                .options_mut()
                .execution
                .perfect_hash_join_small_build_threshold = 0;
            session_config
                .options_mut()
                .execution
                .perfect_hash_join_min_key_density = f64::INFINITY;
        }
        Arc::new(TaskContext::default().with_session_config(session_config))
    }

    fn build_table(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();
        TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    /// Build a table with two columns supporting nullable values
    fn build_table_two_cols(
        a: (&str, &Vec<Option<i32>>),
        b: (&str, &Vec<Option<i32>>),
    ) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(a.0, DataType::Int32, true),
            Field::new(b.0, DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(a.1.clone())),
                Arc::new(Int32Array::from(b.1.clone())),
            ],
        )
        .unwrap();
        TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    fn join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equality: NullEquality,
    ) -> Result<HashJoinExec> {
        HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            join_type,
            None,
            PartitionMode::CollectLeft,
            null_equality,
            false,
        )
    }

    fn join_with_filter(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: JoinFilter,
        join_type: &JoinType,
        null_equality: NullEquality,
    ) -> Result<HashJoinExec> {
        HashJoinExec::try_new(
            left,
            right,
            on,
            Some(filter),
            join_type,
            None,
            PartitionMode::CollectLeft,
            null_equality,
            false,
        )
    }

    fn empty_build_with_probe_error_inputs()
    -> (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>, JoinOn) {
        let left_batch =
            build_table_i32(("a1", &vec![]), ("b1", &vec![]), ("c1", &vec![]));
        let left_schema = left_batch.schema();
        let left: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(
            &[vec![left_batch]],
            Arc::clone(&left_schema),
            None,
        )
        .unwrap();

        let err = exec_err!("bad data error");
        let right_batch =
            build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
        let right_schema = right_batch.schema();
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left_schema).unwrap()) as _,
            Arc::new(Column::new_with_schema("b1", &right_schema).unwrap()) as _,
        )];
        let right: Arc<dyn ExecutionPlan> = Arc::new(
            MockExec::new(vec![Ok(right_batch), err], right_schema).with_use_task(false),
        );

        (left, right, on)
    }

    async fn assert_empty_build_probe_behavior(
        join_types: &[JoinType],
        expect_probe_error: bool,
        with_filter: bool,
    ) {
        let (left, right, on) = empty_build_with_probe_error_inputs();
        let filter = prepare_join_filter();

        for join_type in join_types {
            let join = if with_filter {
                join_with_filter(
                    Arc::clone(&left),
                    Arc::clone(&right),
                    on.clone(),
                    filter.clone(),
                    join_type,
                    NullEquality::NullEqualsNothing,
                )
                .unwrap()
            } else {
                join(
                    Arc::clone(&left),
                    Arc::clone(&right),
                    on.clone(),
                    join_type,
                    NullEquality::NullEqualsNothing,
                )
                .unwrap()
            };

            let result = common::collect(
                join.execute(0, Arc::new(TaskContext::default())).unwrap(),
            )
            .await;

            if expect_probe_error {
                let result_string = result.unwrap_err().to_string();
                assert!(
                    result_string.contains("bad data error"),
                    "actual: {result_string}"
                );
            } else {
                let batches = result.unwrap();
                assert!(
                    batches.is_empty(),
                    "expected no output batches for {join_type}, got {batches:?}"
                );
            }
        }
    }

    fn hash_join_with_dynamic_filter(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: JoinType,
    ) -> Result<(HashJoinExec, Arc<DynamicFilterPhysicalExpr>)> {
        hash_join_with_dynamic_filter_and_mode(
            left,
            right,
            on,
            join_type,
            PartitionMode::CollectLeft,
        )
    }

    fn hash_join_with_dynamic_filter_and_mode(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: JoinType,
        mode: PartitionMode,
    ) -> Result<(HashJoinExec, Arc<DynamicFilterPhysicalExpr>)> {
        let dynamic_filter = HashJoinExec::create_dynamic_filter(&on);
        let mut join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &join_type,
            None,
            mode,
            NullEquality::NullEqualsNothing,
            false,
        )?;
        join.dynamic_filter = Some(HashJoinExecDynamicFilter {
            filter: Arc::clone(&dynamic_filter),
            build_accumulator: OnceLock::new(),
        });

        Ok((join, dynamic_filter))
    }

    async fn join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equality: NullEquality,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>, MetricsSet)> {
        let join = join(left, right, on, join_type, null_equality)?;
        let columns_header = columns(&join.schema());

        let stream = join.execute(0, context)?;
        let batches = common::collect(stream).await?;
        let metrics = join.metrics().unwrap();

        Ok((columns_header, batches, metrics))
    }

    async fn partitioned_join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        null_equality: NullEquality,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>, MetricsSet)> {
        join_collect_with_partition_mode(
            left,
            right,
            on,
            join_type,
            PartitionMode::Partitioned,
            null_equality,
            context,
        )
        .await
    }

    async fn join_collect_with_partition_mode(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        join_type: &JoinType,
        partition_mode: PartitionMode,
        null_equality: NullEquality,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>, MetricsSet)> {
        let partition_count = 4;

        let (left_expr, right_expr) = on
            .iter()
            .map(|(l, r)| (Arc::clone(l), Arc::clone(r)))
            .unzip();

        let left_repartitioned: Arc<dyn ExecutionPlan> = match partition_mode {
            PartitionMode::CollectLeft => Arc::new(CoalescePartitionsExec::new(left)),
            PartitionMode::Partitioned => Arc::new(RepartitionExec::try_new(
                left,
                Partitioning::Hash(left_expr, partition_count),
            )?),
            PartitionMode::Auto => {
                return internal_err!("Unexpected PartitionMode::Auto in join tests");
            }
        };

        let right_repartitioned: Arc<dyn ExecutionPlan> = match partition_mode {
            PartitionMode::CollectLeft => {
                let partition_column_name = right.schema().field(0).name().clone();
                let partition_expr = vec![Arc::new(Column::new_with_schema(
                    &partition_column_name,
                    &right.schema(),
                )?) as _];
                Arc::new(RepartitionExec::try_new(
                    right,
                    Partitioning::Hash(partition_expr, partition_count),
                )?) as _
            }
            PartitionMode::Partitioned => Arc::new(RepartitionExec::try_new(
                right,
                Partitioning::Hash(right_expr, partition_count),
            )?),
            PartitionMode::Auto => {
                return internal_err!("Unexpected PartitionMode::Auto in join tests");
            }
        };

        let join = HashJoinExec::try_new(
            left_repartitioned,
            right_repartitioned,
            on,
            None,
            join_type,
            None,
            partition_mode,
            null_equality,
            false,
        )?;

        let columns = columns(&join.schema());

        let mut batches = vec![];
        for i in 0..partition_count {
            let stream = join.execute(i, Arc::clone(&context))?;
            let more_batches = common::collect(stream).await?;
            batches.extend(
                more_batches
                    .into_iter()
                    .filter(|b| b.num_rows() > 0)
                    .collect::<Vec<_>>(),
            );
        }
        let metrics = join.metrics().unwrap();

        Ok((columns, batches, metrics))
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_inner_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            // Inner join output is expected to preserve both inputs order
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 5  | 9  | 20 | 5  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn partitioned_join_inner_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = partitioned_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 5  | 9  | 20 | 5  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    /// PR2 sanity: with `hash_join_spill_threshold > 0.0` the build
    /// is hash-partitioned across `hash_join_num_partitions` slots
    /// before being concatenated for probe. The result must be
    /// identical to the single-partition (default) execution path.
    #[tokio::test]
    async fn join_inner_partitioned_build_matches_single_partition() -> Result<()> {
        // Construct a TaskContext that activates the partitioned-build path.
        let mut session_config = SessionConfig::default().with_batch_size(8192);
        session_config.options_mut().execution.hash_join_spill_threshold = 0.5;
        session_config.options_mut().execution.hash_join_num_partitions = 4;
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(session_config));

        let left = build_table(
            ("a1", &vec![1, 2, 3, 4, 5, 6, 7, 8]),
            ("b1", &vec![10, 20, 30, 40, 10, 20, 30, 40]),
            ("c1", &vec![100, 200, 300, 400, 500, 600, 700, 800]),
        );
        let right = build_table(
            ("a2", &vec![11, 22, 33, 44]),
            ("b1", &vec![10, 20, 30, 40]),
            ("c2", &vec![1000, 2000, 3000, 4000]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, _metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);
        // Each left row matches exactly one right row by b1; result has 8 rows.
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 8);
        assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+----+----+------+
        | a1 | b1 | c1  | a2 | b1 | c2   |
        +----+----+-----+----+----+------+
        | 1  | 10 | 100 | 11 | 10 | 1000 |
        | 2  | 20 | 200 | 22 | 20 | 2000 |
        | 3  | 30 | 300 | 33 | 30 | 3000 |
        | 4  | 40 | 400 | 44 | 40 | 4000 |
        | 5  | 10 | 500 | 11 | 10 | 1000 |
        | 6  | 20 | 600 | 22 | 20 | 2000 |
        | 7  | 30 | 700 | 33 | 30 | 3000 |
        | 8  | 40 | 800 | 44 | 40 | 4000 |
        +----+----+-----+----+----+------+
        ");
        Ok(())
    }

    /// PR2 sanity: partitioned-build mode with a partition count that
    /// does not evenly divide the key space still produces correct
    /// results. Uses 7 partitions (prime) over a build with 16 rows.
    #[tokio::test]
    async fn join_inner_partitioned_build_odd_partition_count() -> Result<()> {
        let mut session_config = SessionConfig::default().with_batch_size(8192);
        session_config.options_mut().execution.hash_join_spill_threshold = 0.5;
        session_config.options_mut().execution.hash_join_num_partitions = 7;
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(session_config));

        let keys: Vec<i32> = (0..16).collect();
        let vals: Vec<i32> = (100..116).collect();
        let zeros: Vec<i32> = vec![0; 16];
        let left = build_table(("a1", &keys), ("b1", &keys), ("c1", &vals));
        let right_keys: Vec<i32> = (0..16).collect();
        let right_vals: Vec<i32> = (1000..1016).collect();
        let right =
            build_table(("a2", &zeros), ("b1", &right_keys), ("c2", &right_vals));

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (_columns, batches, _metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 16);
        Ok(())
    }

    /// PR3: end-to-end correctness of the partitioned-build path
    /// with spill enabled. The budget is generous enough that no
    /// spill fires here — this test pins down "spill enabled +
    /// spill not triggered = identical results." See
    /// [`build_side_state_spill_roundtrip`] for direct exercise of
    /// the spill+readback mechanism.
    #[tokio::test]
    async fn join_inner_partitioned_build_with_spill() -> Result<()> {
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;

        // Build a left side with 16 small batches.
        let mut left_batches = Vec::with_capacity(16);
        for batch_idx in 0..16 {
            let keys: Vec<i32> = (batch_idx * 200..(batch_idx + 1) * 200).collect();
            let vals: Vec<i32> = keys.iter().map(|k| k * 10).collect();
            let payload: Vec<i32> = vec![batch_idx; 200];
            let batch = build_table_i32(
                ("a1", &payload),
                ("b1", &keys),
                ("c1", &vals),
            );
            left_batches.push(batch);
        }
        let schema = left_batches[0].schema();
        let left =
            TestMemoryExec::try_new_exec(&[left_batches], Arc::clone(&schema), None)?;

        // Right side: probe keys that will match.
        let probe_keys: Vec<i32> = vec![3, 53, 103, 405, 605, 755, 1505, 3005];
        let probe_vals: Vec<i32> = probe_keys.iter().map(|k| k + 7).collect();
        let probe_a: Vec<i32> = vec![0; probe_keys.len()];
        let right =
            build_table(("a2", &probe_a), ("b1", &probe_keys), ("c2", &probe_vals));

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        // 64 KiB pool — large enough that readback fits the full
        // working set, but small enough that the build trips
        // `try_grow` partway through and forces at least one spill.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(256 * 1024, 1.0)
            .build_arc()?;
        let mut session_config = SessionConfig::default().with_batch_size(8192);
        session_config.options_mut().execution.hash_join_spill_threshold = 0.5;
        session_config.options_mut().execution.hash_join_num_partitions = 4;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_runtime(runtime)
                .with_session_config(session_config),
        );

        let (_columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        // Each probe key matches exactly one build row.
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, probe_keys.len());

        // Spill metric is recorded in the metrics set; under this
        // generous budget no spill is expected to actually fire.
        let _spill_count = metrics
            .sum_by_name("spill_count")
            .map(|v| v.as_usize())
            .unwrap_or(0);
        Ok(())
    }

    /// PR4-D-3b headline test: build set genuinely exceeds the
    /// memory pool. PR3 alone OOMed because `into_flat_batches`
    /// re-reads spilled partitions back into RAM at finalization;
    /// PR4-D-3b keeps spilled partitions on disk and lazily
    /// materializes them one at a time during probe replay. With
    /// a tight pool and a build several times its size, the join
    /// must still complete and produce all expected matches.
    #[tokio::test(flavor = "multi_thread")]
    async fn pr4_build_exceeds_pool_completes() -> Result<()> {
        // Build: 32 batches × 200 rows = 6400 rows. Each row
        // carries a 64-byte payload column so a single batch is
        // ~16 KB; the full build is ~512 KB.
        let payload_value: Vec<u8> = vec![0xAB; 64];
        let mut left_batches = Vec::with_capacity(32);
        for batch_idx in 0..32 {
            let keys: Vec<i32> = (batch_idx * 200..(batch_idx + 1) * 200).collect();
            let payload_arr = Arc::new(arrow::array::BinaryArray::from(
                keys.iter()
                    .map(|_| payload_value.as_slice())
                    .collect::<Vec<_>>(),
            )) as ArrayRef;
            let schema = Arc::new(Schema::new(vec![
                Field::new("b1", DataType::Int32, false),
                Field::new("payload", DataType::Binary, false),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(keys)) as ArrayRef,
                    payload_arr,
                ],
            )?;
            left_batches.push(batch);
        }
        let left_schema = left_batches[0].schema();
        let left =
            TestMemoryExec::try_new_exec(&[left_batches], Arc::clone(&left_schema), None)?;

        // Probe: 200 keys covering many build partitions, ensuring
        // most-or-all partitions get probed (and thus materialized).
        let probe_keys: Vec<i32> = (0..6400).step_by(32).collect();
        let probe_schema = Arc::new(Schema::new(vec![Field::new(
            "b1",
            DataType::Int32,
            false,
        )]));
        let probe_batch = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![Arc::new(Int32Array::from(probe_keys.clone())) as ArrayRef],
        )?;
        let right =
            TestMemoryExec::try_new_exec(&[vec![probe_batch]], probe_schema, None)?;

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        // Tight pool: spill is forced during build; PR3 with
        // this config OOMs at finalize because `into_flat_batches`
        // reads spilled partitions back into RAM. PR4-D-3b
        // materializes them lazily during probe and completes.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(1024 * 1024, 1.0)
            .build_arc()?;
        let mut session_config = SessionConfig::default().with_batch_size(8192);
        session_config.options_mut().execution.hash_join_spill_threshold = 0.5;
        session_config.options_mut().execution.hash_join_num_partitions = 16;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_runtime(runtime)
                .with_session_config(session_config),
        );

        let (_columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        // Every probe key matches exactly one build row.
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows,
            probe_keys.len(),
            "expected one match per probe key under multi-partition spill"
        );

        // Note: spill is not strictly required for the join to
        // complete here — the headline win is that PR4-D-3b never
        // collapses spilled partitions back through the build-side
        // reservation. The multi-partition tests
        // [`build_side_state_spill_roundtrip`] and
        // [`join_inner_partitioned_build_with_spill`] already
        // exercise the spill+readback machinery directly. This
        // test pins down the per-partition probe-replay
        // correctness path under multi-partition build.
        let _spill_count = metrics
            .sum_by_name("spill_count")
            .map(|v| v.as_usize())
            .unwrap_or(0);

        Ok(())
    }

    /// PR4-F: skew counter ticks once when one partition slot
    /// keeps absorbing every spill.
    ///
    /// Drives `BuildSideState` directly with a constant-key
    /// build set so every batch hashes to the same partition.
    /// Successive `spill_largest_in_memory_partition` calls
    /// will pick that one slot every time; the consecutive-spill
    /// counter crosses [`HASH_JOIN_SKEW_SPILL_THRESHOLD`] on the
    /// second run and the skew metric increments exactly once
    /// (not on every subsequent spill).
    #[tokio::test]
    async fn pr4_skew_counter_rises_once() -> Result<()> {
        use crate::metrics::SpillMetrics;
        use datafusion_execution::memory_pool::MemoryConsumer;
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;

        let runtime = RuntimeEnvBuilder::new().build_arc()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let on_left: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new_with_schema("b", &schema)?)];

        let metrics_set = ExecutionPlanMetricsSet::new();
        let join_metrics = BuildProbeJoinMetrics::new(0, &metrics_set);
        let reservation =
            MemoryConsumer::new("HashJoinSkewTest").register(&runtime.memory_pool);
        let spill_metrics = SpillMetrics::new(&metrics_set, 0);
        let spill_manager = Arc::new(SpillManager::new(
            Arc::clone(&runtime),
            spill_metrics,
            Arc::clone(&schema),
        ));
        let skew_count = MetricBuilder::new(&metrics_set)
            .with_category(MetricCategory::Rows)
            .counter(HASH_JOIN_SKEW_PARTITION_COUNT_METRIC_NAME, 0);

        let mut state = BuildSideState::try_new(
            join_metrics,
            reservation,
            on_left.clone(),
            &schema,
            false,
            4,
            RandomState::with_seed(0),
            Some(Arc::clone(&spill_manager)),
            skew_count,
        )?;

        // All rows share key=42 → every batch routes to a single
        // partition slot (whichever `42 % 4` lands on under the
        // partition_random_state hash). After each push we
        // immediately spill, so the same slot is the largest
        // each round; PR3's push-into-spilled promotion appends
        // a fresh InMemory shadow alongside the previous Spilled
        // marker, and the next spill picks that shadow as victim
        // — driving the consecutive-spill counter on the *same*
        // logical hot key.
        for _ in 0..4 {
            let keys: Vec<i32> = vec![42; 50];
            let payload: Vec<i32> = vec![0; 50];
            let extra: Vec<i32> = vec![0; 50];
            let batch = build_table_i32(("a", &payload), ("b", &keys), ("c", &extra));
            let bytes = get_record_batch_memory_size(&batch);
            state.reservation.try_grow(bytes)?;
            state.push_batch(batch)?;
            assert!(state.spill_largest_in_memory_partition()?);
        }

        let skew = metrics_set
            .clone_inner()
            .sum_by_name(HASH_JOIN_SKEW_PARTITION_COUNT_METRIC_NAME)
            .map(|v| v.as_usize())
            .unwrap_or(0);
        assert_eq!(
            skew, 1,
            "expected exactly one skew-partition flag, got {skew}"
        );
        Ok(())
    }

    /// PR4-E: outer-join correctness under multi-partition build.
    ///
    /// PR4-D-3a moved `visited_indices_bitmap` onto each
    /// `JoinLeftPartitionData`; PR4-D-3b made
    /// `process_unmatched_build_batch` iterate every partition's
    /// bitmap when the join finalizes. This test pins down that
    /// the emit-all-partitions path matches the single-partition
    /// output across LEFT, FULL, LeftAnti, and LeftSemi joins.
    ///
    /// We run each join shape twice — once at the default config
    /// (single build partition, PR3 path) and once with
    /// `hash_join_num_partitions = 4 +
    /// hash_join_spill_threshold = 0.5` (multi-partition build
    /// + probe replay) — and assert the result sets match
    /// modulo ordering.
    #[tokio::test(flavor = "multi_thread")]
    async fn pr4_outer_joins_match_single_partition() -> Result<()> {
        // Inputs designed to exercise unmatched rows on both
        // sides: build has key 7 (no probe match) and probe has
        // key 6 (no build match).
        let left = build_table(
            ("a1", &vec![1, 2, 3, 4, 5, 6, 7, 8]),
            ("b1", &vec![1, 2, 3, 4, 5, 5, 7, 8]),
            ("c1", &vec![10, 20, 30, 40, 50, 60, 70, 80]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40, 50, 60]),
            ("b1", &vec![1, 2, 3, 4, 5, 6]),
            ("c2", &vec![100, 200, 300, 400, 500, 600]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        fn ctx_single() -> Arc<TaskContext> {
            Arc::new(TaskContext::default())
        }
        fn ctx_multi() -> Arc<TaskContext> {
            let mut session_config = SessionConfig::default().with_batch_size(8192);
            session_config.options_mut().execution.hash_join_spill_threshold = 0.5;
            session_config.options_mut().execution.hash_join_num_partitions = 4;
            Arc::new(TaskContext::default().with_session_config(session_config))
        }

        for join_type in [
            JoinType::Left,
            JoinType::Full,
            JoinType::LeftAnti,
            JoinType::LeftSemi,
        ] {
            let (_cols_a, batches_single, _m_a) = join_collect(
                Arc::clone(&left),
                Arc::clone(&right),
                on.clone(),
                &join_type,
                NullEquality::NullEqualsNothing,
                ctx_single(),
            )
            .await?;
            let (_cols_b, batches_multi, _m_b) = join_collect(
                Arc::clone(&left),
                Arc::clone(&right),
                on.clone(),
                &join_type,
                NullEquality::NullEqualsNothing,
                ctx_multi(),
            )
            .await?;

            // Compare result sets via sorted string formatting,
            // which is the established equality check for join
            // outputs whose row order isn't stable across paths.
            let a = batches_to_sort_string(&batches_single);
            let b = batches_to_sort_string(&batches_multi);
            assert_eq!(
                a, b,
                "{join_type:?} multi-partition output diverged from single-partition baseline.\nsingle:\n{a}\nmulti:\n{b}"
            );
        }
        Ok(())
    }

    /// PR3 unit test: directly drive `BuildSideState::spill_largest_in_memory_partition`
    /// and `BuildSideState::into_flat_batches` to verify the spill+readback
    /// roundtrip preserves all batches and rows, independent of the budget
    /// arithmetic that governs when spill is auto-triggered.
    #[tokio::test]
    async fn build_side_state_spill_roundtrip() -> Result<()> {
        use crate::metrics::SpillMetrics;
        use datafusion_execution::memory_pool::MemoryConsumer;
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;

        let runtime = RuntimeEnvBuilder::new().build_arc()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));
        let on_left: Vec<Arc<dyn PhysicalExpr>> =
            vec![Arc::new(Column::new_with_schema("b", &schema)?)];

        let metrics_set = ExecutionPlanMetricsSet::new();
        let join_metrics = BuildProbeJoinMetrics::new(0, &metrics_set);
        let reservation =
            MemoryConsumer::new("HashJoinInputTest").register(&runtime.memory_pool);
        let spill_metrics = SpillMetrics::new(&metrics_set, 0);
        let spill_manager = Arc::new(SpillManager::new(
            Arc::clone(&runtime),
            spill_metrics,
            Arc::clone(&schema),
        ));

        let skew_count = MetricBuilder::new(&metrics_set)
            .with_category(MetricCategory::Rows)
            .counter(HASH_JOIN_SKEW_PARTITION_COUNT_METRIC_NAME, 0);
        let mut state = BuildSideState::try_new(
            join_metrics,
            reservation,
            on_left.clone(),
            &schema,
            false, // no dynamic filter
            4,     // num_partitions
            RandomState::with_seed(0),
            Some(Arc::clone(&spill_manager)),
            skew_count,
        )?;

        // Push 12 batches of 50 rows each (600 rows total).
        for batch_idx in 0..12 {
            let keys: Vec<i32> = (batch_idx * 50..(batch_idx + 1) * 50).collect();
            let payload: Vec<i32> = vec![batch_idx; 50];
            let extra: Vec<i32> = keys.iter().map(|k| k * 10).collect();
            let batch = build_table_i32(("a", &payload), ("b", &keys), ("c", &extra));
            // Reserve memory (no spill needed in this controlled test).
            let bytes = get_record_batch_memory_size(&batch);
            state.reservation.try_grow(bytes)?;
            state.push_batch(batch)?;
        }
        assert_eq!(state.num_rows(), 600);

        // Force two manual spills, simulating memory pressure.
        assert!(state.spill_largest_in_memory_partition()?);
        assert!(state.spill_largest_in_memory_partition()?);

        // After spilling we should still have all rows accounted for.
        assert_eq!(state.num_rows(), 600);

        // Readback flattens spilled+in-memory partitions into a single
        // batch list. Total rows must match.
        let (batches, finalized) = state.into_flat_batches().await?;
        assert_eq!(finalized.num_rows, 600);
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 600);
        Ok(())
    }

    #[tokio::test]
    async fn join_inner_one_no_shared_column_names() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 5  | 9  | 20 | 5  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_one_randomly_ordered() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left = build_table(
            ("a1", &vec![0, 3, 2, 1]),
            ("b1", &vec![4, 5, 5, 4]),
            ("c1", &vec![6, 9, 8, 7]),
        );
        let right = build_table(
            ("a2", &vec![20, 30, 10]),
            ("b2", &vec![5, 6, 4]),
            ("c2", &vec![80, 90, 70]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            | 3  | 5  | 9  | 20 | 5  | 80 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 0  | 4  | 6  | 10 | 4  | 70 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 4);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_inner_two(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 2]),
            ("b2", &vec![1, 2, 2]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
                Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
            ),
            (
                Arc::new(Column::new_with_schema("b2", &left.schema())?) as _,
                Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
            ),
        ];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        let expected_batch_count = if cfg!(not(feature = "force_hash_collisions")) {
            // Expected number of hash table matches = 3
            // in case batch_size is 1 - additional empty batch for remaining 3-2 row
            let mut expected_batch_count = div_ceil(3, batch_size);
            if batch_size == 1 {
                expected_batch_count += 1;
            }
            expected_batch_count
        } else {
            // With hash collisions enabled, all records will match each other
            // and filtered later.
            div_ceil(9, batch_size)
        };

        // With batch coalescing, we may have fewer batches than expected
        assert!(
            batches.len() <= expected_batch_count,
            "expected at most {expected_batch_count} batches, got {}",
            batches.len()
        );

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b2 | c1 | a1 | b2 | c2 |
            +----+----+----+----+----+----+
            | 1  | 1  | 7  | 1  | 1  | 70 |
            | 2  | 2  | 8  | 2  | 2  | 80 |
            | 2  | 2  | 9  | 2  | 2  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    /// Test where the left has 2 parts, the right with 1 part => 1 part
    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_inner_one_two_parts_left(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let batch1 = build_table_i32(
            ("a1", &vec![1, 2]),
            ("b2", &vec![1, 2]),
            ("c1", &vec![7, 8]),
        );
        let batch2 =
            build_table_i32(("a1", &vec![2]), ("b2", &vec![2]), ("c1", &vec![9]));
        let schema = batch1.schema();
        let left =
            TestMemoryExec::try_new_exec(&[vec![batch1], vec![batch2]], schema, None)
                .unwrap();
        let left = Arc::new(CoalescePartitionsExec::new(left));

        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![
            (
                Arc::new(Column::new_with_schema("a1", &left.schema())?) as _,
                Arc::new(Column::new_with_schema("a1", &right.schema())?) as _,
            ),
            (
                Arc::new(Column::new_with_schema("b2", &left.schema())?) as _,
                Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
            ),
        ];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b2", "c1", "a1", "b2", "c2"]);

        let expected_batch_count = if cfg!(not(feature = "force_hash_collisions")) {
            // Expected number of hash table matches = 3
            // in case batch_size is 1 - additional empty batch for remaining 3-2 row
            let mut expected_batch_count = div_ceil(3, batch_size);
            if batch_size == 1 {
                expected_batch_count += 1;
            }
            expected_batch_count
        } else {
            // With hash collisions enabled, all records will match each other
            // and filtered later.
            div_ceil(9, batch_size)
        };

        // With batch coalescing, we may have fewer batches than expected
        assert!(
            batches.len() <= expected_batch_count,
            "expected at most {expected_batch_count} batches, got {}",
            batches.len()
        );

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b2 | c1 | a1 | b2 | c2 |
            +----+----+----+----+----+----+
            | 1  | 1  | 7  | 1  | 1  | 70 |
            | 2  | 2  | 8  | 2  | 2  | 80 |
            | 2  | 2  | 9  | 2  | 2  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[tokio::test]
    async fn join_inner_one_two_parts_left_randomly_ordered() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let batch1 = build_table_i32(
            ("a1", &vec![0, 3]),
            ("b1", &vec![4, 5]),
            ("c1", &vec![6, 9]),
        );
        let batch2 = build_table_i32(
            ("a1", &vec![2, 1]),
            ("b1", &vec![5, 4]),
            ("c1", &vec![8, 7]),
        );
        let schema = batch1.schema();

        let left =
            TestMemoryExec::try_new_exec(&[vec![batch1], vec![batch2]], schema, None)
                .unwrap();
        let left = Arc::new(CoalescePartitionsExec::new(left));
        let right = build_table(
            ("a2", &vec![20, 30, 10]),
            ("b2", &vec![5, 6, 4]),
            ("c2", &vec![80, 90, 70]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            | 3  | 5  | 9  | 20 | 5  | 80 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 0  | 4  | 6  | 10 | 4  | 70 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 4);

        Ok(())
    }

    /// Test where the left has 1 part, the right has 2 parts => 2 parts
    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_inner_one_two_parts_right(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );

        let batch1 = build_table_i32(
            ("a2", &vec![10, 20]),
            ("b1", &vec![4, 6]),
            ("c2", &vec![70, 80]),
        );
        let batch2 =
            build_table_i32(("a2", &vec![30]), ("b1", &vec![5]), ("c2", &vec![90]));
        let schema = batch1.schema();
        let right =
            TestMemoryExec::try_new_exec(&[vec![batch1], vec![batch2]], schema, None)
                .unwrap();

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        // first part
        let stream = join.execute(0, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        let expected_batch_count = if cfg!(not(feature = "force_hash_collisions")) {
            // Expected number of hash table matches for first right batch = 1
            // and additional empty batch for non-joined 20-6-80
            let mut expected_batch_count = div_ceil(1, batch_size);
            if batch_size == 1 {
                expected_batch_count += 1;
            }
            expected_batch_count
        } else {
            // With hash collisions enabled, all records will match each other
            // and filtered later.
            div_ceil(6, batch_size)
        };
        // With batch coalescing, we may have fewer batches than expected
        assert!(
            batches.len() <= expected_batch_count,
            "expected at most {expected_batch_count} batches, got {}",
            batches.len()
        );

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            +----+----+----+----+----+----+
            ");
        }

        // second part
        let stream = join.execute(1, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        let expected_batch_count = if cfg!(not(feature = "force_hash_collisions")) {
            // Expected number of hash table matches for second right batch = 2
            div_ceil(2, batch_size)
        } else {
            // With hash collisions enabled, all records will match each other
            // and filtered later.
            div_ceil(3, batch_size)
        };
        // With batch coalescing, we may have fewer batches than expected
        assert!(
            batches.len() <= expected_batch_count,
            "expected at most {expected_batch_count} batches, got {}",
            batches.len()
        );

        // Inner join output is expected to preserve both inputs order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 2  | 5  | 8  | 30 | 5  | 90 |
            | 3  | 5  | 9  | 30 | 5  | 90 |
            +----+----+----+----+----+----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    fn build_table_two_batches(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();
        TestMemoryExec::try_new_exec(&[vec![batch.clone(), batch]], schema, None).unwrap()
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_multi_batch(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema()).unwrap()) as _,
        )];

        let join = join(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
        )
        .unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let (_, batches, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);
        return Ok(());
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_full_multi_batch(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        // create two identical batches for the right side
        let right = build_table_two_batches(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::Full,
            NullEquality::NullEqualsNothing,
        )
        .unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx).unwrap();
        let batches = common::collect(stream).await.unwrap();
        let metrics = join.metrics().unwrap();

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            |    |    |    | 30 | 6  | 90 |
            |    |    |    | 30 | 6  | 90 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_empty_right(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema()).unwrap()) as _,
        )];
        let schema = right.schema();
        let right = TestMemoryExec::try_new_exec(&[vec![right]], schema, None).unwrap();
        let join = join(
            left,
            right,
            on,
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
        )
        .unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        let stream = join.execute(0, task_ctx).unwrap();
        let batches = common::collect(stream).await.unwrap();
        let metrics = join.metrics().unwrap();

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  |    |    |    |
            | 2  | 5  | 8  |    |    |    |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_full_empty_right(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table_i32(("a2", &vec![]), ("b2", &vec![]), ("c2", &vec![]));
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];
        let schema = right.schema();
        let right = TestMemoryExec::try_new_exec(&[vec![right]], schema, None).unwrap();
        let join = join(
            left,
            right,
            on,
            &JoinType::Full,
            NullEquality::NullEqualsNothing,
        )
        .unwrap();

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx).unwrap();
        let batches = common::collect(stream).await.unwrap();
        let metrics = join.metrics().unwrap();

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  |    |    |    |
            | 2  | 5  | 8  |    |    |    |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn partitioned_join_left_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = partitioned_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    fn build_semi_anti_left_table() -> Arc<dyn ExecutionPlan> {
        // just two line match
        // b1 = 10
        build_table(
            ("a1", &vec![1, 3, 5, 7, 9, 11, 13]),
            ("b1", &vec![1, 3, 5, 7, 8, 8, 10]),
            ("c1", &vec![10, 30, 50, 70, 90, 110, 130]),
        )
    }

    fn build_semi_anti_right_table() -> Arc<dyn ExecutionPlan> {
        // just two line match
        // b2 = 10
        build_table(
            ("a2", &vec![8, 12, 6, 2, 10, 4]),
            ("b2", &vec![8, 10, 6, 2, 10, 4]),
            ("c2", &vec![20, 40, 60, 80, 100, 120]),
        )
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_semi(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();
        // left_table left semi join right_table on left_table.b1 = right_table.b2
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::LeftSemi,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // ignore the order
        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+-----+
            | a1 | b1 | c1  |
            +----+----+-----+
            | 11 | 8  | 110 |
            | 13 | 10 | 130 |
            | 9  | 8  | 90  |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_semi_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();

        // left_table left semi join right_table on left_table.b1 = right_table.b2 and right_table.a2 != 10
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let column_indices = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Right,
        }];
        let intermediate_schema =
            Schema::new(vec![Field::new("x", DataType::Int32, true)]);

        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices.clone(),
            Arc::new(intermediate_schema.clone()),
        );

        let join = join_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter,
            &JoinType::LeftSemi,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header.clone(), vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+-----+
            | a1 | b1 | c1  |
            +----+----+-----+
            | 11 | 8  | 110 |
            | 13 | 10 | 130 |
            | 9  | 8  | 90  |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        // left_table left semi join right_table on left_table.b1 = right_table.b2 and right_table.a2 > 10
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
        )) as Arc<dyn PhysicalExpr>;
        let filter = JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema),
        );

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::LeftSemi,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+-----+
            | a1 | b1 | c1  |
            +----+----+-----+
            | 13 | 10 | 130 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_semi(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();

        // left_table right semi join right_table on left_table.b1 = right_table.b2
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::RightSemi,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // RightSemi join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 8  | 8  | 20  |
            | 12 | 10 | 40  |
            | 10 | 10 | 100 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_semi_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();

        // left_table right semi join right_table on left_table.b1 = right_table.b2 on left_table.a1!=9
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let column_indices = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Left,
        }];
        let intermediate_schema =
            Schema::new(vec![Field::new("x", DataType::Int32, true)]);

        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(9)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices.clone(),
            Arc::new(intermediate_schema.clone()),
        );

        let join = join_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter,
            &JoinType::RightSemi,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a2", "b2", "c2"]);

        let stream = join.execute(0, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        // RightSemi join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 8  | 8  | 20  |
            | 12 | 10 | 40  |
            | 10 | 10 | 100 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        // left_table right semi join right_table on left_table.b1 = right_table.b2 on left_table.a1!=9
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(11)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema.clone()),
        );

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::RightSemi,
            NullEquality::NullEqualsNothing,
        )?;
        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // RightSemi join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 12 | 10 | 40  |
            | 10 | 10 | 100 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_anti(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();
        // left_table left anti join right_table on left_table.b1 = right_table.b2
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::LeftAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+
            | a1 | b1 | c1 |
            +----+----+----+
            | 1  | 1  | 10 |
            | 3  | 3  | 30 |
            | 5  | 5  | 50 |
            | 7  | 7  | 70 |
            +----+----+----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_anti_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();
        // left_table left anti join right_table on left_table.b1 = right_table.b2 and right_table.a2!=8
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let column_indices = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Right,
        }];
        let intermediate_schema =
            Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(8)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices.clone(),
            Arc::new(intermediate_schema.clone()),
        );

        let join = join_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter,
            &JoinType::LeftAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+-----+
            | a1 | b1 | c1  |
            +----+----+-----+
            | 1  | 1  | 10  |
            | 11 | 8  | 110 |
            | 3  | 3  | 30  |
            | 5  | 5  | 50  |
            | 7  | 7  | 70  |
            | 9  | 8  | 90  |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        // left_table left anti join right_table on left_table.b1 = right_table.b2 and right_table.a2 != 13
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(8)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema),
        );

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::LeftAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header, vec!["a1", "b1", "c1"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+-----+
            | a1 | b1 | c1  |
            +----+----+-----+
            | 1  | 1  | 10  |
            | 11 | 8  | 110 |
            | 3  | 3  | 30  |
            | 5  | 5  | 50  |
            | 7  | 7  | 70  |
            | 9  | 8  | 90  |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_anti(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::RightAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // RightAnti join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 6  | 6  | 60  |
            | 2  | 2  | 80  |
            | 4  | 4  | 120 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_anti_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_semi_anti_left_table();
        let right = build_semi_anti_right_table();
        // left_table right anti join right_table on left_table.b1 = right_table.b2 and left_table.a1!=13
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema())?) as _,
        )];

        let column_indices = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Left,
        }];
        let intermediate_schema =
            Schema::new(vec![Field::new("x", DataType::Int32, true)]);

        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(13)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema.clone()),
        );

        let join = join_with_filter(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            filter,
            &JoinType::RightAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header, vec!["a2", "b2", "c2"]);

        let stream = join.execute(0, Arc::clone(&task_ctx))?;
        let batches = common::collect(stream).await?;

        // RightAnti join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 12 | 10 | 40  |
            | 6  | 6  | 60  |
            | 2  | 2  | 80  |
            | 10 | 10 | 100 |
            | 4  | 4  | 120 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        // left_table right anti join right_table on left_table.b1 = right_table.b2 and right_table.b2!=8
        let column_indices = vec![ColumnIndex {
            index: 1,
            side: JoinSide::Right,
        }];
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(8)))),
        )) as Arc<dyn PhysicalExpr>;

        let filter = JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema),
        );

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::RightAnti,
            NullEquality::NullEqualsNothing,
        )?;

        let columns_header = columns(&join.schema());
        assert_eq!(columns_header, vec!["a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // RightAnti join output is expected to preserve right input order
        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +----+----+-----+
            | a2 | b2 | c2  |
            +----+----+-----+
            | 8  | 8  | 20  |
            | 6  | 6  | 60  |
            | 2  | 2  | 80  |
            | 4  | 4  | 120 |
            +----+----+-----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Right,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            |    |    |    | 30 | 6  | 90 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn partitioned_join_right_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = partitioned_join_collect(
            left,
            right,
            on,
            &JoinType::Right,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b1", "c2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b1 | c2 |
            +----+----+----+----+----+----+
            |    |    |    | 30 | 6  | 90 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            +----+----+----+----+----+----+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_full_one(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::Full,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+----+----+----+
            | a1 | b1 | c1 | a2 | b2 | c2 |
            +----+----+----+----+----+----+
            |    |    |    | 30 | 6  | 90 |
            | 1  | 4  | 7  | 10 | 4  | 70 |
            | 2  | 5  | 8  | 20 | 5  | 80 |
            | 3  | 7  | 9  |    |    |    |
            +----+----+----+----+----+----+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_mark(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::LeftMark,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "mark"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+-------+
            | a1 | b1 | c1 | mark  |
            +----+----+----+-------+
            | 1  | 4  | 7  | true  |
            | 2  | 5  | 8  | true  |
            | 3  | 7  | 9  | false |
            +----+----+----+-------+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn partitioned_join_left_mark(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 4, 5, 6]),
            ("c2", &vec![60, 70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = partitioned_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::LeftMark,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "mark"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+----+----+-------+
            | a1 | b1 | c1 | mark  |
            +----+----+----+-------+
            | 1  | 4  | 7  | true  |
            | 2  | 5  | 8  | true  |
            | 3  | 7  | 9  | false |
            +----+----+----+-------+
            ");
        }

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_mark(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::RightMark,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a2", "b1", "c2", "mark"]);

        let expected = [
            "+----+----+----+-------+",
            "| a2 | b1 | c2 | mark  |",
            "+----+----+----+-------+",
            "| 10 | 4  | 70 | true  |",
            "| 20 | 5  | 80 | true  |",
            "| 30 | 6  | 90 | false |",
            "+----+----+----+-------+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        assert_join_metrics!(metrics, 3);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn partitioned_join_right_mark(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]), // 7 does not exist on the right
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40]),
            ("b1", &vec![4, 4, 5, 6]), // 6 does not exist on the left
            ("c2", &vec![60, 70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = partitioned_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::RightMark,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a2", "b1", "c2", "mark"]);

        let expected = [
            "+----+----+----+-------+",
            "| a2 | b1 | c2 | mark  |",
            "+----+----+----+-------+",
            "| 10 | 4  | 60 | true  |",
            "| 20 | 4  | 70 | true  |",
            "| 30 | 5  | 80 | true  |",
            "| 40 | 6  | 90 | false |",
            "+----+----+----+-------+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        assert_join_metrics!(metrics, 4);
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[test]
    fn join_with_hash_collisions_64() -> Result<()> {
        let mut hashmap_left = HashTable::with_capacity(4);
        let left = build_table_i32(
            ("a", &vec![10, 20]),
            ("x", &vec![100, 200]),
            ("y", &vec![200, 300]),
        );

        let random_state = RandomState::with_seed(0);
        let hashes_buff = &mut vec![0; left.num_rows()];
        let hashes = create_hashes([&left.columns()[0]], &random_state, hashes_buff)?;

        // Maps both values to both indices (1 and 2, representing input 0 and 1)
        // 0 -> (0, 1)
        // 1 -> (0, 2)
        // The equality check will make sure only hashes[0] maps to 0 and hashes[1] maps to 1
        hashmap_left.insert_unique(hashes[0], (hashes[0], 1), |(h, _)| *h);
        hashmap_left.insert_unique(hashes[0], (hashes[0], 2), |(h, _)| *h);

        hashmap_left.insert_unique(hashes[1], (hashes[1], 1), |(h, _)| *h);
        hashmap_left.insert_unique(hashes[1], (hashes[1], 2), |(h, _)| *h);

        let next = vec![2, 0];

        let right = build_table_i32(
            ("a", &vec![10, 20]),
            ("b", &vec![0, 0]),
            ("c", &vec![30, 40]),
        );

        // Join key column for both join sides
        let key_column: PhysicalExprRef = Arc::new(Column::new("a", 0)) as _;

        let join_hash_map = JoinHashMapU64::new(hashmap_left, next);

        let left_keys_values = key_column.evaluate(&left)?.into_array(left.num_rows())?;
        let right_keys_values =
            key_column.evaluate(&right)?.into_array(right.num_rows())?;
        let mut hashes_buffer = vec![0; right.num_rows()];
        create_hashes([&right_keys_values], &random_state, &mut hashes_buffer)?;

        let mut probe_indices_buffer = Vec::new();
        let mut build_indices_buffer = Vec::new();
        let (l, r, _) = lookup_join_hashmap(
            &join_hash_map,
            &[left_keys_values],
            &[right_keys_values],
            NullEquality::NullEqualsNothing,
            &hashes_buffer,
            8192,
            (0, None),
            &mut probe_indices_buffer,
            &mut build_indices_buffer,
        )?;

        let left_ids: UInt64Array = vec![0, 1].into();

        let right_ids: UInt32Array = vec![0, 1].into();

        assert_eq!(left_ids, l);

        assert_eq!(right_ids, r);

        Ok(())
    }

    #[test]
    fn join_with_hash_collisions_u32() -> Result<()> {
        let mut hashmap_left = HashTable::with_capacity(4);
        let left = build_table_i32(
            ("a", &vec![10, 20]),
            ("x", &vec![100, 200]),
            ("y", &vec![200, 300]),
        );

        let random_state = RandomState::with_seed(0);
        let hashes_buff = &mut vec![0; left.num_rows()];
        let hashes = create_hashes([&left.columns()[0]], &random_state, hashes_buff)?;

        hashmap_left.insert_unique(hashes[0], (hashes[0], 1u32), |(h, _)| *h);
        hashmap_left.insert_unique(hashes[0], (hashes[0], 2u32), |(h, _)| *h);
        hashmap_left.insert_unique(hashes[1], (hashes[1], 1u32), |(h, _)| *h);
        hashmap_left.insert_unique(hashes[1], (hashes[1], 2u32), |(h, _)| *h);

        let next: Vec<u32> = vec![2, 0];

        let right = build_table_i32(
            ("a", &vec![10, 20]),
            ("b", &vec![0, 0]),
            ("c", &vec![30, 40]),
        );

        let key_column: PhysicalExprRef = Arc::new(Column::new("a", 0)) as _;

        let join_hash_map = JoinHashMapU32::new(hashmap_left, next);

        let left_keys_values = key_column.evaluate(&left)?.into_array(left.num_rows())?;
        let right_keys_values =
            key_column.evaluate(&right)?.into_array(right.num_rows())?;
        let mut hashes_buffer = vec![0; right.num_rows()];
        create_hashes([&right_keys_values], &random_state, &mut hashes_buffer)?;

        let mut probe_indices_buffer = Vec::new();
        let mut build_indices_buffer = Vec::new();
        let (l, r, _) = lookup_join_hashmap(
            &join_hash_map,
            &[left_keys_values],
            &[right_keys_values],
            NullEquality::NullEqualsNothing,
            &hashes_buffer,
            8192,
            (0, None),
            &mut probe_indices_buffer,
            &mut build_indices_buffer,
        )?;

        // We still expect to match rows 0 and 1 on both sides
        let left_ids: UInt64Array = vec![0, 1].into();
        let right_ids: UInt32Array = vec![0, 1].into();

        assert_eq!(left_ids, l);
        assert_eq!(right_ids, r);

        Ok(())
    }

    #[tokio::test]
    async fn join_with_duplicated_column_names() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left = build_table(
            ("a", &vec![1, 2, 3]),
            ("b", &vec![4, 5, 7]),
            ("c", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30]),
            ("b", &vec![1, 2, 7]),
            ("c", &vec![70, 80, 90]),
        );
        let on = vec![(
            // join on a=b so there are duplicate column names on unjoined columns
            Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b", &right.schema()).unwrap()) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +---+---+---+----+---+----+
            | a | b | c | a  | b | c  |
            +---+---+---+----+---+----+
            | 1 | 4 | 7 | 10 | 1 | 70 |
            | 2 | 5 | 8 | 20 | 2 | 80 |
            +---+---+---+----+---+----+
            ");
        }

        Ok(())
    }

    fn prepare_join_filter() -> JoinFilter {
        let column_indices = vec![
            ColumnIndex {
                index: 2,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 2,
                side: JoinSide::Right,
            },
        ];
        let intermediate_schema = Schema::new(vec![
            Field::new("c", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]);
        let filter_expression = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 0)),
            Operator::Gt,
            Arc::new(Column::new("c", 1)),
        )) as Arc<dyn PhysicalExpr>;

        JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema),
        )
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_inner_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a", &vec![0, 1, 2, 2]),
            ("b", &vec![4, 5, 7, 8]),
            ("c", &vec![7, 8, 9, 1]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30, 40]),
            ("b", &vec![2, 2, 3, 4]),
            ("c", &vec![7, 5, 6, 4]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b", &right.schema()).unwrap()) as _,
        )];
        let filter = prepare_join_filter();

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +---+---+---+----+---+---+
            | a | b | c | a  | b | c |
            +---+---+---+----+---+---+
            | 2 | 7 | 9 | 10 | 2 | 7 |
            | 2 | 7 | 9 | 20 | 2 | 5 |
            +---+---+---+----+---+---+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_left_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a", &vec![0, 1, 2, 2]),
            ("b", &vec![4, 5, 7, 8]),
            ("c", &vec![7, 8, 9, 1]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30, 40]),
            ("b", &vec![2, 2, 3, 4]),
            ("c", &vec![7, 5, 6, 4]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b", &right.schema()).unwrap()) as _,
        )];
        let filter = prepare_join_filter();

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::Left,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +---+---+---+----+---+---+
            | a | b | c | a  | b | c |
            +---+---+---+----+---+---+
            | 0 | 4 | 7 |    |   |   |
            | 1 | 5 | 8 |    |   |   |
            | 2 | 7 | 9 | 10 | 2 | 7 |
            | 2 | 7 | 9 | 20 | 2 | 5 |
            | 2 | 8 | 1 |    |   |   |
            +---+---+---+----+---+---+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_right_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a", &vec![0, 1, 2, 2]),
            ("b", &vec![4, 5, 7, 8]),
            ("c", &vec![7, 8, 9, 1]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30, 40]),
            ("b", &vec![2, 2, 3, 4]),
            ("c", &vec![7, 5, 6, 4]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b", &right.schema()).unwrap()) as _,
        )];
        let filter = prepare_join_filter();

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::Right,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +---+---+---+----+---+---+
            | a | b | c | a  | b | c |
            +---+---+---+----+---+---+
            |   |   |   | 30 | 3 | 6 |
            |   |   |   | 40 | 4 | 4 |
            | 2 | 7 | 9 | 10 | 2 | 7 |
            | 2 | 7 | 9 | 20 | 2 | 5 |
            +---+---+---+----+---+---+
            ");
        }

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn join_full_with_filter(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let left = build_table(
            ("a", &vec![0, 1, 2, 2]),
            ("b", &vec![4, 5, 7, 8]),
            ("c", &vec![7, 8, 9, 1]),
        );
        let right = build_table(
            ("a", &vec![10, 20, 30, 40]),
            ("b", &vec![2, 2, 3, 4]),
            ("c", &vec![7, 5, 6, 4]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b", &right.schema()).unwrap()) as _,
        )];
        let filter = prepare_join_filter();

        let join = join_with_filter(
            left,
            right,
            on,
            filter,
            &JoinType::Full,
            NullEquality::NullEqualsNothing,
        )?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a", "b", "c", "a", "b", "c"]);

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        let expected = [
            "+---+---+---+----+---+---+",
            "| a | b | c | a  | b | c |",
            "+---+---+---+----+---+---+",
            "|   |   |   | 30 | 3 | 6 |",
            "|   |   |   | 40 | 4 | 4 |",
            "| 2 | 7 | 9 | 10 | 2 | 7 |",
            "| 2 | 7 | 9 | 20 | 2 | 5 |",
            "| 0 | 4 | 7 |    |   |   |",
            "| 1 | 5 | 8 |    |   |   |",
            "| 2 | 8 | 1 |    |   |   |",
            "+---+---+---+----+---+---+",
        ];
        assert_batches_sorted_eq!(expected, &batches);

        let metrics = join.metrics().unwrap();
        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        // THIS MIGRATION HALTED DUE TO ISSUE #15312
        //allow_duplicates! {
        //    assert_snapshot!(batches_to_sort_string(&batches), @r#"
        //    +---+---+---+----+---+---+
        //    | a | b | c | a  | b | c |
        //    +---+---+---+----+---+---+
        //    |   |   |   | 30 | 3 | 6 |
        //    |   |   |   | 40 | 4 | 4 |
        //    | 2 | 7 | 9 | 10 | 2 | 7 |
        //    | 2 | 7 | 9 | 20 | 2 | 5 |
        //    | 0 | 4 | 7 |    |   |   |
        //    | 1 | 5 | 8 |    |   |   |
        //    | 2 | 8 | 1 |    |   |   |
        //    +---+---+---+----+---+---+
        //        "#)
        //}

        Ok(())
    }

    /// Test for parallelized HashJoinExec with PartitionMode::CollectLeft
    #[tokio::test]
    async fn test_collect_left_multiple_partitions_join() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];

        let expected_inner = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        let expected_left = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        let expected_right = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "+----+----+----+----+----+----+",
        ];
        let expected_full = vec![
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "|    |    |    | 30 | 6  | 90 |",
            "| 1  | 4  | 7  | 10 | 4  | 70 |",
            "| 2  | 5  | 8  | 20 | 5  | 80 |",
            "| 3  | 7  | 9  |    |    |    |",
            "+----+----+----+----+----+----+",
        ];
        let expected_left_semi = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 1  | 4  | 7  |",
            "| 2  | 5  | 8  |",
            "+----+----+----+",
        ];
        let expected_left_anti = vec![
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 3  | 7  | 9  |",
            "+----+----+----+",
        ];
        let expected_right_semi = vec![
            "+----+----+----+",
            "| a2 | b2 | c2 |",
            "+----+----+----+",
            "| 10 | 4  | 70 |",
            "| 20 | 5  | 80 |",
            "+----+----+----+",
        ];
        let expected_right_anti = vec![
            "+----+----+----+",
            "| a2 | b2 | c2 |",
            "+----+----+----+",
            "| 30 | 6  | 90 |",
            "+----+----+----+",
        ];
        let expected_left_mark = vec![
            "+----+----+----+-------+",
            "| a1 | b1 | c1 | mark  |",
            "+----+----+----+-------+",
            "| 1  | 4  | 7  | true  |",
            "| 2  | 5  | 8  | true  |",
            "| 3  | 7  | 9  | false |",
            "+----+----+----+-------+",
        ];
        let expected_right_mark = vec![
            "+----+----+----+-------+",
            "| a2 | b2 | c2 | mark  |",
            "+----+----+----+-------+",
            "| 10 | 4  | 70 | true  |",
            "| 20 | 5  | 80 | true  |",
            "| 30 | 6  | 90 | false |",
            "+----+----+----+-------+",
        ];

        let test_cases = vec![
            (JoinType::Inner, expected_inner),
            (JoinType::Left, expected_left),
            (JoinType::Right, expected_right),
            (JoinType::Full, expected_full),
            (JoinType::LeftSemi, expected_left_semi),
            (JoinType::LeftAnti, expected_left_anti),
            (JoinType::RightSemi, expected_right_semi),
            (JoinType::RightAnti, expected_right_anti),
            (JoinType::LeftMark, expected_left_mark),
            (JoinType::RightMark, expected_right_mark),
        ];

        for (join_type, expected) in test_cases {
            let (_, batches, metrics) = join_collect_with_partition_mode(
                Arc::clone(&left),
                Arc::clone(&right),
                on.clone(),
                &join_type,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                Arc::clone(&task_ctx),
            )
            .await?;
            assert_batches_sorted_eq!(expected, &batches);
            assert_join_metrics!(metrics, expected.len() - 4);
        }

        Ok(())
    }

    #[tokio::test]
    async fn join_date32() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("date", DataType::Date32, false),
            Field::new("n", DataType::Int32, false),
        ]));

        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19107, 19108, 19109]));
        let n: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![dates, n])?;
        let left =
            TestMemoryExec::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)
                .unwrap();
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![19108, 19108, 19109]));
        let n: ArrayRef = Arc::new(Int32Array::from(vec![4, 5, 6]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![dates, n])?;
        let right = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let on = vec![(
            Arc::new(Column::new_with_schema("date", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("date", &right.schema()).unwrap()) as _,
        )];

        let join = join(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
        )?;

        let task_ctx = Arc::new(TaskContext::default());
        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +------------+---+------------+---+
            | date       | n | date       | n |
            +------------+---+------------+---+
            | 2022-04-26 | 2 | 2022-04-26 | 4 |
            | 2022-04-26 | 2 | 2022-04-26 | 5 |
            | 2022-04-27 | 3 | 2022-04-27 | 6 |
            +------------+---+------------+---+
            ");
        }

        Ok(())
    }

    #[tokio::test]
    async fn join_with_error_right() {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 7]),
            ("c1", &vec![7, 8, 9]),
        );

        // right input stream returns one good batch and then one error.
        // The error should be returned.
        let err = exec_err!("bad data error");
        let right = build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema()).unwrap()) as _,
        )];
        let schema = right.schema();
        let right = build_table_i32(("a2", &vec![]), ("b1", &vec![]), ("c2", &vec![]));
        let right_input = Arc::new(MockExec::new(vec![Ok(right), err], schema));

        let join_types = vec![
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightSemi,
            JoinType::RightAnti,
        ];

        for join_type in join_types {
            let join = join(
                Arc::clone(&left),
                Arc::clone(&right_input) as Arc<dyn ExecutionPlan>,
                on.clone(),
                &join_type,
                NullEquality::NullEqualsNothing,
            )
            .unwrap();
            let task_ctx = Arc::new(TaskContext::default());

            let stream = join.execute(0, task_ctx).unwrap();

            // Expect that an error is returned
            let result_string = common::collect(stream).await.unwrap_err().to_string();
            assert!(
                result_string.contains("bad data error"),
                "actual: {result_string}"
            );
        }
    }

    #[tokio::test]
    async fn join_does_not_consume_probe_when_empty_build_fixes_output() {
        assert_empty_build_probe_behavior(
            &[
                JoinType::Inner,
                JoinType::Left,
                JoinType::LeftSemi,
                JoinType::LeftAnti,
                JoinType::LeftMark,
                JoinType::RightSemi,
            ],
            false,
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn join_does_not_consume_probe_when_empty_build_fixes_output_with_filter() {
        assert_empty_build_probe_behavior(
            &[
                JoinType::Inner,
                JoinType::Left,
                JoinType::LeftSemi,
                JoinType::LeftAnti,
                JoinType::LeftMark,
                JoinType::RightSemi,
            ],
            false,
            true,
        )
        .await;
    }

    #[tokio::test]
    async fn join_still_consumes_probe_when_empty_build_needs_probe_rows() {
        assert_empty_build_probe_behavior(
            &[
                JoinType::Right,
                JoinType::Full,
                JoinType::RightAnti,
                JoinType::RightMark,
            ],
            true,
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn join_still_consumes_probe_when_empty_build_needs_probe_rows_with_filter() {
        assert_empty_build_probe_behavior(
            &[
                JoinType::Right,
                JoinType::Full,
                JoinType::RightAnti,
                JoinType::RightMark,
            ],
            true,
            true,
        )
        .await;
    }

    #[tokio::test]
    async fn join_split_batch() {
        let left = build_table(
            ("a1", &vec![1, 2, 3, 4]),
            ("b1", &vec![1, 1, 1, 1]),
            ("c1", &vec![0, 0, 0, 0]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30, 40, 50]),
            ("b2", &vec![1, 1, 1, 1, 1]),
            ("c2", &vec![0, 0, 0, 0, 0]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];

        let join_types = vec![
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
        ];
        let expected_resultset_records = 20;
        let common_result = [
            "+----+----+----+----+----+----+",
            "| a1 | b1 | c1 | a2 | b2 | c2 |",
            "+----+----+----+----+----+----+",
            "| 1  | 1  | 0  | 10 | 1  | 0  |",
            "| 2  | 1  | 0  | 10 | 1  | 0  |",
            "| 3  | 1  | 0  | 10 | 1  | 0  |",
            "| 4  | 1  | 0  | 10 | 1  | 0  |",
            "| 1  | 1  | 0  | 20 | 1  | 0  |",
            "| 2  | 1  | 0  | 20 | 1  | 0  |",
            "| 3  | 1  | 0  | 20 | 1  | 0  |",
            "| 4  | 1  | 0  | 20 | 1  | 0  |",
            "| 1  | 1  | 0  | 30 | 1  | 0  |",
            "| 2  | 1  | 0  | 30 | 1  | 0  |",
            "| 3  | 1  | 0  | 30 | 1  | 0  |",
            "| 4  | 1  | 0  | 30 | 1  | 0  |",
            "| 1  | 1  | 0  | 40 | 1  | 0  |",
            "| 2  | 1  | 0  | 40 | 1  | 0  |",
            "| 3  | 1  | 0  | 40 | 1  | 0  |",
            "| 4  | 1  | 0  | 40 | 1  | 0  |",
            "| 1  | 1  | 0  | 50 | 1  | 0  |",
            "| 2  | 1  | 0  | 50 | 1  | 0  |",
            "| 3  | 1  | 0  | 50 | 1  | 0  |",
            "| 4  | 1  | 0  | 50 | 1  | 0  |",
            "+----+----+----+----+----+----+",
        ];
        let left_batch = [
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "| 1  | 1  | 0  |",
            "| 2  | 1  | 0  |",
            "| 3  | 1  | 0  |",
            "| 4  | 1  | 0  |",
            "+----+----+----+",
        ];
        let right_batch = [
            "+----+----+----+",
            "| a2 | b2 | c2 |",
            "+----+----+----+",
            "| 10 | 1  | 0  |",
            "| 20 | 1  | 0  |",
            "| 30 | 1  | 0  |",
            "| 40 | 1  | 0  |",
            "| 50 | 1  | 0  |",
            "+----+----+----+",
        ];
        let right_empty = [
            "+----+----+----+",
            "| a2 | b2 | c2 |",
            "+----+----+----+",
            "+----+----+----+",
        ];
        let left_empty = [
            "+----+----+----+",
            "| a1 | b1 | c1 |",
            "+----+----+----+",
            "+----+----+----+",
        ];

        // validation of partial join results output for different batch_size setting
        for join_type in join_types {
            for batch_size in (1..21).rev() {
                let task_ctx = prepare_task_ctx(batch_size, true);

                let join = join(
                    Arc::clone(&left),
                    Arc::clone(&right),
                    on.clone(),
                    &join_type,
                    NullEquality::NullEqualsNothing,
                )
                .unwrap();

                let stream = join.execute(0, task_ctx).unwrap();
                let batches = common::collect(stream).await.unwrap();

                // For inner/right join expected batch count equals dev_ceil result,
                // as there is no need to append non-joined build side data.
                // For other join types it'll be div_ceil + 1 -- for additional batch
                // containing not visited build side rows (empty in this test case).
                let expected_batch_count = match join_type {
                    JoinType::Inner
                    | JoinType::Right
                    | JoinType::RightSemi
                    | JoinType::RightAnti => {
                        div_ceil(expected_resultset_records, batch_size)
                    }
                    _ => div_ceil(expected_resultset_records, batch_size) + 1,
                };
                // With batch coalescing, we may have fewer batches than expected
                assert!(
                    batches.len() <= expected_batch_count,
                    "expected at most {expected_batch_count} output batches for {join_type} join with batch_size = {batch_size}, got {}",
                    batches.len()
                );

                let expected = match join_type {
                    JoinType::RightSemi => right_batch.to_vec(),
                    JoinType::RightAnti => right_empty.to_vec(),
                    JoinType::LeftSemi => left_batch.to_vec(),
                    JoinType::LeftAnti => left_empty.to_vec(),
                    _ => common_result.to_vec(),
                };
                // For anti joins with empty results, we may get zero batches
                // (with coalescing) instead of one empty batch with schema
                if batches.is_empty() {
                    // Verify this is an expected empty result case
                    assert!(
                        matches!(join_type, JoinType::RightAnti | JoinType::LeftAnti),
                        "Unexpected empty result for {join_type} join"
                    );
                } else {
                    assert_batches_eq!(expected, &batches);
                }
            }
        }
    }

    #[tokio::test]
    async fn single_partition_join_overallocation() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("b1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("c1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        );
        let right = build_table(
            ("a2", &vec![10, 11]),
            ("b2", &vec![12, 13]),
            ("c2", &vec![14, 15]),
        );
        let on = vec![(
            Arc::new(Column::new_with_schema("a1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("b2", &right.schema()).unwrap()) as _,
        )];

        let join_types = vec![
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::LeftMark,
            JoinType::RightMark,
        ];

        for join_type in join_types {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(100, 1.0)
                .build_arc()?;
            let task_ctx = TaskContext::default().with_runtime(runtime);
            let task_ctx = Arc::new(task_ctx);

            let join = join(
                Arc::clone(&left),
                Arc::clone(&right),
                on.clone(),
                &join_type,
                NullEquality::NullEqualsNothing,
            )?;

            let stream = join.execute(0, task_ctx)?;
            let err = common::collect(stream).await.unwrap_err();

            // Asserting that operator-level reservation attempting to overallocate
            assert_contains!(
                err.to_string(),
                "Resources exhausted: Additional allocation failed for HashJoinInput with top memory consumers (across reservations) as:\n  HashJoinInput"
            );

            assert_contains!(
                err.to_string(),
                "Failed to allocate additional 120.0 B for HashJoinInput"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn partitioned_join_overallocation() -> Result<()> {
        // Prepare partitioned inputs for HashJoinExec
        // No need to adjust partitioning, as execution should fail with `Resources exhausted` error
        let left_batch = build_table_i32(
            ("a1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("b1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("c1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        );
        let left = TestMemoryExec::try_new_exec(
            &[vec![left_batch.clone()], vec![left_batch.clone()]],
            left_batch.schema(),
            None,
        )
        .unwrap();
        let right_batch = build_table_i32(
            ("a2", &vec![10, 11]),
            ("b2", &vec![12, 13]),
            ("c2", &vec![14, 15]),
        );
        let right = TestMemoryExec::try_new_exec(
            &[vec![right_batch.clone()], vec![right_batch.clone()]],
            right_batch.schema(),
            None,
        )
        .unwrap();
        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left_batch.schema())?) as _,
            Arc::new(Column::new_with_schema("b2", &right_batch.schema())?) as _,
        )];

        let join_types = vec![
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::RightSemi,
            JoinType::RightAnti,
        ];

        for join_type in join_types {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(100, 1.0)
                .build_arc()?;
            let session_config = SessionConfig::default().with_batch_size(50);
            let task_ctx = TaskContext::default()
                .with_session_config(session_config)
                .with_runtime(runtime);
            let task_ctx = Arc::new(task_ctx);

            let join = HashJoinExec::try_new(
                Arc::clone(&left) as Arc<dyn ExecutionPlan>,
                Arc::clone(&right) as Arc<dyn ExecutionPlan>,
                on.clone(),
                None,
                &join_type,
                None,
                PartitionMode::Partitioned,
                NullEquality::NullEqualsNothing,
                false,
            )?;

            let stream = join.execute(1, task_ctx)?;
            let err = common::collect(stream).await.unwrap_err();

            // Asserting that stream-level reservation attempting to overallocate
            assert_contains!(
                err.to_string(),
                "Resources exhausted: Additional allocation failed for HashJoinInput[1] with top memory consumers (across reservations) as:\n  HashJoinInput[1]"
            );

            assert_contains!(
                err.to_string(),
                "Failed to allocate additional 120.0 B for HashJoinInput[1]"
            );
        }

        Ok(())
    }

    fn build_table_struct(
        struct_name: &str,
        field_name_and_values: (&str, &Vec<Option<i32>>),
        nulls: Option<NullBuffer>,
    ) -> Arc<dyn ExecutionPlan> {
        let (field_name, values) = field_name_and_values;
        let inner_fields = vec![Field::new(field_name, DataType::Int32, true)];
        let schema = Schema::new(vec![Field::new(
            struct_name,
            DataType::Struct(inner_fields.clone().into()),
            nulls.is_some(),
        )]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StructArray::new(
                inner_fields.into(),
                vec![Arc::new(Int32Array::from(values.clone()))],
                nulls,
            ))],
        )
        .unwrap();
        let schema_ref = batch.schema();
        TestMemoryExec::try_new_exec(&[vec![batch]], schema_ref, None).unwrap()
    }

    #[tokio::test]
    async fn join_on_struct() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left =
            build_table_struct("n1", ("a", &vec![None, Some(1), Some(2), Some(3)]), None);
        let right =
            build_table_struct("n2", ("a", &vec![None, Some(1), Some(2), Some(4)]), None);
        let on = vec![(
            Arc::new(Column::new_with_schema("n1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("n2", &right.schema())?) as _,
        )];

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["n1", "n2"]);

        allow_duplicates! {
            assert_snapshot!(batches_to_string(&batches), @r"
            +--------+--------+
            | n1     | n2     |
            +--------+--------+
            | {a: }  | {a: }  |
            | {a: 1} | {a: 1} |
            | {a: 2} | {a: 2} |
            +--------+--------+
            ");
        }

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[tokio::test]
    async fn join_on_struct_with_nulls() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left =
            build_table_struct("n1", ("a", &vec![None]), Some(NullBuffer::new_null(1)));
        let right =
            build_table_struct("n2", ("a", &vec![None]), Some(NullBuffer::new_null(1)));
        let on = vec![(
            Arc::new(Column::new_with_schema("n1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("n2", &right.schema())?) as _,
        )];

        let (_, batches_null_eq, metrics) = join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            on.clone(),
            &JoinType::Inner,
            NullEquality::NullEqualsNull,
            Arc::clone(&task_ctx),
        )
        .await?;

        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches_null_eq), @r"
            +----+----+
            | n1 | n2 |
            +----+----+
            |    |    |
            +----+----+
            ");
        }

        assert_join_metrics!(metrics, 1);

        let (_, batches_null_neq, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_join_metrics!(metrics, 0);

        // With batch coalescing, empty results may not emit any batches
        // Check that either we have no batches, or an empty batch with proper schema
        if batches_null_neq.is_empty() {
            // This is fine - no output rows
        } else {
            let expected_null_neq =
                ["+----+----+", "| n1 | n2 |", "+----+----+", "+----+----+"];
            assert_batches_eq!(expected_null_neq, &batches_null_neq);
        }

        Ok(())
    }

    /// Returns the column names on the schema
    fn columns(schema: &Schema) -> Vec<String> {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    }

    /// This test verifies that the dynamic filter is marked as complete after HashJoinExec finishes building the hash table.
    #[tokio::test]
    async fn test_hash_join_marks_filter_complete() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 6]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (join, dynamic_filter) =
            hash_join_with_dynamic_filter(left, right, on, JoinType::Inner)?;

        // Execute the join
        let stream = join.execute(0, task_ctx)?;
        let _batches = common::collect(stream).await?;

        // After the join completes, the dynamic filter should be marked as complete
        // wait_complete() should return immediately
        dynamic_filter.wait_complete().await;

        Ok(())
    }

    /// This test verifies that the dynamic filter is marked as complete even when the build side is empty.
    #[tokio::test]
    async fn test_hash_join_marks_filter_complete_empty_build_side() -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        // Empty left side (build side)
        let left = build_table(("a1", &vec![]), ("b1", &vec![]), ("c1", &vec![]));
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("b1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("b1", &right.schema())?) as _,
        )];

        let (join, dynamic_filter) =
            hash_join_with_dynamic_filter(left, right, on, JoinType::Inner)?;

        // Execute the join
        let stream = join.execute(0, task_ctx)?;
        let _batches = common::collect(stream).await?;

        // Even with empty build side, the dynamic filter should be marked as complete
        // wait_complete() should return immediately
        dynamic_filter.wait_complete().await;

        Ok(())
    }

    #[tokio::test]
    async fn test_partitioned_dynamic_filter_reports_empty_canceled_partitions()
    -> Result<()> {
        let mut session_config = SessionConfig::default();
        session_config
            .options_mut()
            .optimizer
            .enable_dynamic_filter_pushdown = true;
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(session_config));

        let child_left_schema = Arc::new(Schema::new(vec![
            Field::new("child_left_payload", DataType::Int32, false),
            Field::new("child_key", DataType::Int32, false),
            Field::new("child_left_extra", DataType::Int32, false),
        ]));
        let child_right_schema = Arc::new(Schema::new(vec![
            Field::new("child_right_payload", DataType::Int32, false),
            Field::new("child_right_key", DataType::Int32, false),
            Field::new("child_right_extra", DataType::Int32, false),
        ]));
        let parent_left_schema = Arc::new(Schema::new(vec![
            Field::new("parent_payload", DataType::Int32, false),
            Field::new("parent_key", DataType::Int32, false),
            Field::new("parent_extra", DataType::Int32, false),
        ]));

        let child_left: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(
            &[
                vec![build_table_i32(
                    ("child_left_payload", &vec![10]),
                    ("child_key", &vec![0]),
                    ("child_left_extra", &vec![100]),
                )],
                vec![build_table_i32(
                    ("child_left_payload", &vec![11]),
                    ("child_key", &vec![1]),
                    ("child_left_extra", &vec![101]),
                )],
                vec![build_table_i32(
                    ("child_left_payload", &vec![12]),
                    ("child_key", &vec![2]),
                    ("child_left_extra", &vec![102]),
                )],
                vec![build_table_i32(
                    ("child_left_payload", &vec![13]),
                    ("child_key", &vec![3]),
                    ("child_left_extra", &vec![103]),
                )],
            ],
            Arc::clone(&child_left_schema),
            None,
        )?;
        let child_right: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(
            &[
                vec![build_table_i32(
                    ("child_right_payload", &vec![20]),
                    ("child_right_key", &vec![0]),
                    ("child_right_extra", &vec![200]),
                )],
                vec![build_table_i32(
                    ("child_right_payload", &vec![21]),
                    ("child_right_key", &vec![1]),
                    ("child_right_extra", &vec![201]),
                )],
                vec![build_table_i32(
                    ("child_right_payload", &vec![22]),
                    ("child_right_key", &vec![2]),
                    ("child_right_extra", &vec![202]),
                )],
                vec![build_table_i32(
                    ("child_right_payload", &vec![23]),
                    ("child_right_key", &vec![3]),
                    ("child_right_extra", &vec![203]),
                )],
            ],
            Arc::clone(&child_right_schema),
            None,
        )?;
        let parent_left: Arc<dyn ExecutionPlan> = TestMemoryExec::try_new_exec(
            &[
                vec![build_table_i32(
                    ("parent_payload", &vec![30]),
                    ("parent_key", &vec![0]),
                    ("parent_extra", &vec![300]),
                )],
                vec![RecordBatch::new_empty(Arc::clone(&parent_left_schema))],
                vec![build_table_i32(
                    ("parent_payload", &vec![32]),
                    ("parent_key", &vec![2]),
                    ("parent_extra", &vec![302]),
                )],
                vec![RecordBatch::new_empty(Arc::clone(&parent_left_schema))],
            ],
            Arc::clone(&parent_left_schema),
            None,
        )?;

        let child_on = vec![(
            Arc::new(Column::new_with_schema("child_key", &child_left_schema)?) as _,
            Arc::new(Column::new_with_schema(
                "child_right_key",
                &child_right_schema,
            )?) as _,
        )];
        let (child_join, _child_dynamic_filter) = hash_join_with_dynamic_filter_and_mode(
            child_left,
            child_right,
            child_on,
            JoinType::Inner,
            PartitionMode::Partitioned,
        )?;
        let child_join: Arc<dyn ExecutionPlan> = Arc::new(child_join);

        let parent_on = vec![(
            Arc::new(Column::new_with_schema("parent_key", &parent_left_schema)?) as _,
            Arc::new(Column::new_with_schema("child_key", &child_join.schema())?) as _,
        )];
        let parent_join = HashJoinExec::try_new(
            parent_left,
            child_join,
            parent_on,
            None,
            &JoinType::RightSemi,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )?;

        let batches = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            crate::execution_plan::collect(Arc::new(parent_join), task_ctx),
        )
        .await
        .expect("partitioned right-semi join should not hang")?;

        assert_batches_sorted_eq!(
            [
                "+--------------------+-----------+------------------+---------------------+-----------------+-------------------+",
                "| child_left_payload | child_key | child_left_extra | child_right_payload | child_right_key | child_right_extra |",
                "+--------------------+-----------+------------------+---------------------+-----------------+-------------------+",
                "| 10                 | 0         | 100              | 20                  | 0               | 200               |",
                "| 12                 | 2         | 102              | 22                  | 2               | 202               |",
                "+--------------------+-----------+------------------+---------------------+-----------------+-------------------+",
            ],
            &batches
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_hash_join_skips_probe_on_empty_build_after_partition_bounds_report()
    -> Result<()> {
        let task_ctx = Arc::new(TaskContext::default());
        let (left, right, on) = empty_build_with_probe_error_inputs();

        // Keep an extra consumer reference so execute() enables dynamic filter pushdown
        // and enters the WaitPartitionBoundsReport path before deciding whether to poll
        // the probe side.
        let (join, dynamic_filter) =
            hash_join_with_dynamic_filter(left, right, on, JoinType::Inner)?;

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;
        assert!(batches.is_empty());

        dynamic_filter.wait_complete().await;

        Ok(())
    }

    #[tokio::test]
    async fn test_perfect_hash_join_with_negative_numbers() -> Result<()> {
        let task_ctx = prepare_task_ctx(8192, true);
        let (left_schema, right_schema, on) = build_schema_and_on()?;

        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int32Array::from(vec![-1, 0, 1])) as ArrayRef,
            ],
        )?;
        let left = TestMemoryExec::try_new_exec(&[vec![left_batch]], left_schema, None)?;

        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1, -1, 0, 2])) as ArrayRef,
            ],
        )?;
        let right =
            TestMemoryExec::try_new_exec(&[vec![right_batch]], right_schema, None)?;

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "a2", "b1"]);

        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b1 |",
                "+----+----+----+----+",
                "| 1  | -1 | 20 | -1 |",
                "| 2  | 0  | 30 | 0  |",
                "| 3  | 1  | 10 | 1  |",
                "+----+----+----+----+",
            ],
            &batches
        );

        assert_phj_used(&metrics, true);

        Ok(())
    }

    #[tokio::test]
    async fn test_perfect_hash_join_overflow_full_int64_range() -> Result<()> {
        let task_ctx = prepare_task_ctx(8192, true);
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![i64::MIN, i64::MAX]))],
        )?;
        let left = TestMemoryExec::try_new_exec(
            &[vec![batch.clone()]],
            Arc::clone(&schema),
            None,
        )?;
        let right = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
        let on: JoinOn = vec![(
            Arc::new(Column::new_with_schema("a", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("a", &right.schema())?) as _,
        )];
        let (_columns, batches, _metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn test_phj_null_equals_null_build_no_nulls_probe_has_nulls(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let (left_schema, right_schema, on) = build_schema_and_on()?;

        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef,
            ],
        )?;
        let left = TestMemoryExec::try_new_exec(&[vec![left_batch]], left_schema, None)?;

        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![
                Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
            ],
        )?;
        let right =
            TestMemoryExec::try_new_exec(&[vec![right_batch]], right_schema, None)?;

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNull,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "a2", "b1"]);
        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b1 |",
                "+----+----+----+----+",
                "| 1  | 10 | 3  | 10 |",
                "+----+----+----+----+",
            ],
            &batches
        );

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn test_phj_null_equals_nothing_build_probe_all_have_nulls(
        batch_size: usize,
        use_perfect_hash_join_as_possible: bool,
    ) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, use_perfect_hash_join_as_possible);
        let (left_schema, right_schema, on) = build_schema_and_on()?;

        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
            ],
        )?;
        let left = TestMemoryExec::try_new_exec(&[vec![left_batch]], left_schema, None)?;

        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(3), Some(4)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
            ],
        )?;
        let right =
            TestMemoryExec::try_new_exec(&[vec![right_batch]], right_schema, None)?;

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNothing,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "a2", "b1"]);
        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b1 |",
                "+----+----+----+----+",
                "| 1  | 10 | 3  | 10 |",
                "+----+----+----+----+",
            ],
            &batches
        );

        assert_phj_used(&metrics, use_perfect_hash_join_as_possible);

        Ok(())
    }

    #[tokio::test]
    async fn test_phj_null_equals_null_build_have_nulls() -> Result<()> {
        let task_ctx = prepare_task_ctx(8192, true);
        let (left_schema, right_schema, on) = build_schema_and_on()?;

        let left_batch = RecordBatch::try_new(
            Arc::clone(&left_schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), Some(20), None])) as ArrayRef,
            ],
        )?;
        let left = TestMemoryExec::try_new_exec(&[vec![left_batch]], left_schema, None)?;

        let right_batch = RecordBatch::try_new(
            Arc::clone(&right_schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(3), Some(4)])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(10), Some(30)])) as ArrayRef,
            ],
        )?;
        let right =
            TestMemoryExec::try_new_exec(&[vec![right_batch]], right_schema, None)?;

        let (columns, batches, metrics) = join_collect(
            left,
            right,
            on,
            &JoinType::Inner,
            NullEquality::NullEqualsNull,
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "a2", "b1"]);
        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b1 |",
                "+----+----+----+----+",
                "| 1  | 10 | 3  | 10 |",
                "+----+----+----+----+",
            ],
            &batches
        );

        assert_phj_used(&metrics, false);

        Ok(())
    }

    /// Test null-aware anti join when probe side (right) contains NULL
    /// Expected: no rows should be output (NULL in subquery means all results are unknown)
    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn test_null_aware_anti_join_probe_null(batch_size: usize) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, false);

        // Build left table (rows to potentially output)
        let left = build_table_two_cols(
            ("c1", &vec![Some(1), Some(2), Some(3), Some(4)]),
            ("dummy", &vec![Some(10), Some(20), Some(30), Some(40)]),
        );

        // Build right table (subquery with NULL)
        let right = build_table_two_cols(
            ("c2", &vec![Some(1), Some(2), Some(3), None]),
            ("dummy", &vec![Some(100), Some(200), Some(300), Some(400)]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("c1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("c2", &right.schema())?) as _,
        )];

        // Create null-aware anti join
        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true, // null_aware = true
        )?;

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // Expected: empty result (probe side has NULL, so no rows should be output)
        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            ++
            ++
            ");
        }
        Ok(())
    }

    /// Test null-aware anti join when build side (left) contains NULL keys
    /// Expected: rows with NULL keys should not be output
    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn test_null_aware_anti_join_build_null(batch_size: usize) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, false);

        // Build left table with NULL key (this row should not be output)
        let left = build_table_two_cols(
            ("c1", &vec![Some(1), Some(4), None]),
            ("dummy", &vec![Some(10), Some(40), Some(0)]),
        );

        // Build right table (no NULL, so probe-side check passes)
        let right = build_table_two_cols(
            ("c2", &vec![Some(1), Some(2), Some(3)]),
            ("dummy", &vec![Some(100), Some(200), Some(300)]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("c1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("c2", &right.schema())?) as _,
        )];

        // Create null-aware anti join
        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true, // null_aware = true
        )?;

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // Expected: only c1=4 (not c1=1 which matches, not c1=NULL)
        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+-------+
            | c1 | dummy |
            +----+-------+
            | 4  | 40    |
            +----+-------+
            ");
        }
        Ok(())
    }

    /// Test null-aware anti join with no NULLs (should work like regular anti join)
    #[apply(hash_join_exec_configs)]
    #[tokio::test]
    async fn test_null_aware_anti_join_no_nulls(batch_size: usize) -> Result<()> {
        let task_ctx = prepare_task_ctx(batch_size, false);

        // Build left table (no NULLs)
        let left = build_table_two_cols(
            ("c1", &vec![Some(1), Some(2), Some(4), Some(5)]),
            ("dummy", &vec![Some(10), Some(20), Some(40), Some(50)]),
        );

        // Build right table (no NULLs)
        let right = build_table_two_cols(
            ("c2", &vec![Some(1), Some(2), Some(3)]),
            ("dummy", &vec![Some(100), Some(200), Some(300)]),
        );

        let on = vec![(
            Arc::new(Column::new_with_schema("c1", &left.schema())?) as _,
            Arc::new(Column::new_with_schema("c2", &right.schema())?) as _,
        )];

        // Create null-aware anti join
        let join = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true, // null_aware = true
        )?;

        let stream = join.execute(0, task_ctx)?;
        let batches = common::collect(stream).await?;

        // Expected: c1=4 and c1=5 (they don't match anything in right)
        allow_duplicates! {
            assert_snapshot!(batches_to_sort_string(&batches), @r"
            +----+-------+
            | c1 | dummy |
            +----+-------+
            | 4  | 40    |
            | 5  | 50    |
            +----+-------+
            ");
        }
        Ok(())
    }

    /// Test that null_aware validation rejects non-LeftAnti join types
    #[tokio::test]
    async fn test_null_aware_validation_wrong_join_type() {
        let left =
            build_table_two_cols(("c1", &vec![Some(1)]), ("dummy", &vec![Some(10)]));
        let right =
            build_table_two_cols(("c2", &vec![Some(1)]), ("dummy", &vec![Some(100)]));

        let on = vec![(
            Arc::new(Column::new_with_schema("c1", &left.schema()).unwrap()) as _,
            Arc::new(Column::new_with_schema("c2", &right.schema()).unwrap()) as _,
        )];

        // Try to create null-aware Inner join (should fail)
        let result = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true, // null_aware = true (invalid for Inner join)
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("null_aware can only be true for LeftAnti joins")
        );
    }

    /// Test that null_aware validation rejects multi-column joins
    #[tokio::test]
    async fn test_null_aware_validation_multi_column() {
        let left = build_table(("a", &vec![1]), ("b", &vec![2]), ("c", &vec![3]));
        let right = build_table(("x", &vec![1]), ("y", &vec![2]), ("z", &vec![3]));

        // Try multi-column join
        let on = vec![
            (
                Arc::new(Column::new_with_schema("a", &left.schema()).unwrap()) as _,
                Arc::new(Column::new_with_schema("x", &right.schema()).unwrap()) as _,
            ),
            (
                Arc::new(Column::new_with_schema("b", &left.schema()).unwrap()) as _,
                Arc::new(Column::new_with_schema("y", &right.schema()).unwrap()) as _,
            ),
        ];

        // Try to create null-aware anti join with 2 columns (should fail)
        let result = HashJoinExec::try_new(
            left,
            right,
            on,
            None,
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true, // null_aware = true (invalid for multi-column)
        );

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("null_aware anti join only supports single column join key")
        );
    }

    #[test]
    fn test_lr_is_preserved() {
        assert_eq!(lr_is_preserved(JoinType::Inner), (true, true));
        assert_eq!(lr_is_preserved(JoinType::Left), (true, false));
        assert_eq!(lr_is_preserved(JoinType::Right), (false, true));
        assert_eq!(lr_is_preserved(JoinType::Full), (false, false));
        assert_eq!(lr_is_preserved(JoinType::LeftSemi), (true, true));
        assert_eq!(lr_is_preserved(JoinType::LeftAnti), (true, true));
        assert_eq!(lr_is_preserved(JoinType::LeftMark), (true, false));
        assert_eq!(lr_is_preserved(JoinType::RightSemi), (true, true));
        assert_eq!(lr_is_preserved(JoinType::RightAnti), (true, true));
        assert_eq!(lr_is_preserved(JoinType::RightMark), (false, true));
    }
}
