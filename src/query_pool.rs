use std::ops::RangeBounds;

use crate::{Device, HasDevice};
use ash::{prelude::VkResult, vk};

pub struct QueryPool {
    device: Device,
    pool: vk::QueryPool,
    ty: vk::QueryType,
    count: u32,
}

impl HasDevice for QueryPool {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl QueryPool {
    pub fn new(device: Device, query_type: vk::QueryType, query_count: u32) -> VkResult<Self> {
        assert_ne!(
            query_type,
            vk::QueryType::PIPELINE_STATISTICS,
            "Use QueryPool::new_pipeline_statistics instead."
        );
        assert_ne!(
            query_type,
            vk::QueryType::PERFORMANCE_QUERY_KHR,
            "Use QueryPool::new_pipeline_statistics instead."
        );
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo {
                    query_type,
                    query_count,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            pool,
            ty: query_type,
            count: query_count,
        })
    }

    pub fn new_pipeline_statistics(
        device: Device,
        pipeline_statistics: vk::QueryPipelineStatisticFlags,
        query_count: u32,
    ) -> VkResult<Self> {
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo {
                    query_type: vk::QueryType::PIPELINE_STATISTICS,
                    query_count,
                    pipeline_statistics,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            pool,
            ty: vk::QueryType::PIPELINE_STATISTICS,
            count: query_count,
        })
    }

    pub fn new_performance_query(
        device: Device,
        queue_family_index: u32,
        counter_indices: &[u32],
        query_count: u32,
    ) -> VkResult<Self> {
        let pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo {
                    query_type: vk::QueryType::PERFORMANCE_QUERY_KHR,
                    query_count,
                    ..Default::default()
                }
                .push_next(
                    &mut vk::QueryPoolPerformanceCreateInfoKHR::default()
                        .queue_family_index(queue_family_index)
                        .counter_indices(counter_indices),
                ),
                None,
            )?
        };
        Ok(Self {
            device,
            pool,
            ty: vk::QueryType::PIPELINE_STATISTICS,
            count: query_count,
        })
    }

    pub fn raw(&self) -> vk::QueryPool {
        self.pool
    }

    pub fn query_type(&self) -> vk::QueryType {
        self.ty
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    unsafe fn get_results_raw<T>(
        &self,
        first_query: u32,
        flags: vk::QueryResultFlags,
        data: &mut [T],
    ) -> VkResult<()> {
        self.device
            .get_query_pool_results(self.pool, first_query, data, flags)
    }

    pub fn get_results(&self, first_query: u32, data: &mut [u32]) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe { self.get_results_raw(first_query, vk::QueryResultFlags::empty(), data) }
    }
    pub fn get_results_u64(&self, first_query: u32, data: &mut [u64]) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe { self.get_results_raw(first_query, vk::QueryResultFlags::TYPE_64, data) }
    }
    pub fn get_results_with_availability(
        &self,
        first_query: u32,
        data: &mut [(u32, vk::Bool32)],
    ) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe { self.get_results_raw(first_query, vk::QueryResultFlags::WITH_AVAILABILITY, data) }
    }
    pub fn get_results_with_status(
        &self,
        first_query: u32,
        data: &mut [(u32, vk::QueryResultStatusKHR)],
    ) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe { self.get_results_raw(first_query, vk::QueryResultFlags::WITH_STATUS_KHR, data) }
    }
    pub fn get_results_with_status_u64(
        &self,
        first_query: u32,
        data: &mut [(u64, QueryPoolStatus64)],
    ) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe {
            self.get_results_raw(
                first_query,
                vk::QueryResultFlags::WITH_STATUS_KHR | vk::QueryResultFlags::TYPE_64,
                data,
            )
        }
    }
    pub fn get_results_with_availability_u64(
        &self,
        first_query: u32,
        data: &mut [(u64, QueryPoolAvailability64)],
    ) -> VkResult<()> {
        debug_assert_ne!(self.ty, vk::QueryType::PERFORMANCE_QUERY_KHR);
        unsafe {
            self.get_results_raw(
                first_query,
                vk::QueryResultFlags::WITH_AVAILABILITY | vk::QueryResultFlags::TYPE_64,
                data,
            )
        }
    }

    pub fn reset(&mut self, range: impl RangeBounds<u32>) {
        let first_query = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let last_query = match range.end_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n - 1,
            std::ops::Bound::Unbounded => self.count - 1,
        };
        unsafe {
            self.device
                .reset_query_pool(self.pool, first_query, last_query + 1 - first_query);
        }
    }
}

pub type QueryPoolStatus64 = u64;
pub type QueryPoolAvailability64 = u64;
