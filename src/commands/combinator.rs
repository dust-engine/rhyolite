use std::{
    fmt::Debug,
    future::Future,
    pin::Pin,
    ptr::Pointee,
    task::{Context, Poll},
};

use crate::future::GPUFutureBlockReturnValue;

use crate::future::GPUFutureBlock;

pub struct Zip<A: GPUFutureBlock, B: GPUFutureBlock>(A, B);
impl<A: GPUFutureBlock, B: GPUFutureBlock> Zip<A, B> {
    fn a(self: Pin<&mut Self>) -> Pin<&mut A> {
        unsafe { self.map_unchecked_mut(|s| &mut s.0) }
    }
    fn b(self: Pin<&mut Self>) -> Pin<&mut B> {
        unsafe { self.map_unchecked_mut(|s| &mut s.1) }
    }
}
impl<A: GPUFutureBlock, B: GPUFutureBlock> Future for Zip<A, B> {
    type Output = GPUFutureBlockReturnValue<(A::Returned, B::Returned), (A::Retained, B::Retained)>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let a = self.as_mut().a().poll(cx);
        let b = self.as_mut().b().poll(cx);
        match (a, b) {
            (
                Poll::Ready(GPUFutureBlockReturnValue {
                    output: a_output,
                    retained_values: a_retained_value,
                }),
                Poll::Ready(GPUFutureBlockReturnValue {
                    output: b_output,
                    retained_values: b_retained_value,
                }),
            ) => {
                return Poll::Ready(GPUFutureBlockReturnValue {
                    output: (a_output, b_output),
                    retained_values: (a_retained_value, b_retained_value),
                });
            }
            (Poll::Pending, Poll::Pending) => {
                return Poll::Pending;
            }
            _ => {
                panic!()
            }
        }
    }
}
pub fn zip<A: GPUFutureBlock, B: GPUFutureBlock>(a: A, b: B) -> Zip<A, B> {
    Zip(a, b)
}

/// A container for dropping items.
/// It stores the item size, item data, and item drop fn ptr inline in the same buffer.
/// This allows us to avoid reallocations when running the same pointer multiple times.
#[derive(Default)]
pub struct DynRetainedValueContainer {
    data: Vec<u64>,
}
impl DynRetainedValueContainer {
    pub fn push<T>(&mut self, item: T) {
        let drop_fn: unsafe fn(*mut T) = std::ptr::drop_in_place::<T> as unsafe fn(_);
        let drop_fn_address: usize = unsafe { std::mem::transmute(drop_fn) };
        let obj_size = std::mem::size_of::<T>().div_ceil(8); // in number of u64s
        self.data.push(drop_fn_address as u64);
        self.data.push(obj_size as u64);
        self.data.reserve(obj_size);
        unsafe {
            // move obj into self.data
            let tail = self.data.as_mut_ptr().add(self.data.len());
            std::ptr::copy_nonoverlapping(&item, tail as *mut T, 1);
            std::mem::forget(item);
            self.data.set_len(self.data.len() + obj_size);
        }
    }
    pub fn drop_in_place(&mut self) {
        let mut i = 0;
        while i < self.data.len() {
            let drop_fn: unsafe fn(*mut u64) =
                unsafe { std::mem::transmute(self.data[i] as usize) };
            i += 1;
            let size = self.data[i]; // in number of u64s
            i += 1;
            let obj_ptr = &mut self.data[i] as *mut u64;
            i += size as usize;
            unsafe {
                (drop_fn)(obj_ptr);
            }
        }
        self.data.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
impl Drop for DynRetainedValueContainer {
    fn drop(&mut self) {
        self.drop_in_place();
    }
}

/// A vector of Box<T> where T is unsized.
/// The vector is specifically optimized for cases such that elements will be pushed into the vector
/// repeatedly. In those cases, no new allocations will be made on subsequent pushes.
/// For example,
pub struct ReusingBoxVec<T: ?Sized>(Vec<*mut T>);
impl<T: ?Sized> Default for ReusingBoxVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}
impl<T: ?Sized + Debug> Debug for ReusingBoxVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReusingBoxVec")?;
        let mut list = f.debug_list();
        for &i in self.0.iter() {
            unsafe {
                let r: &T = &*i;
                list.entry(&r);
            }
        }
        for i in self.0.len()..self.0.capacity() {
            unsafe {
                let ptr = *self.0.as_ptr().add(i);
                if ptr.is_null() {
                    break;
                }
                list.entry_with(|f| f.write_str("uninit"));
            }
        }
        list.finish()
    }
}
impl<T: ?Sized> ReusingBoxVec<T> {
    fn reserve_one(&mut self) {
        let old_capacity = self.0.capacity();
        self.0.reserve(1);
        let new_capacity = self.0.capacity();
        if old_capacity != new_capacity {
            // the memory was grown. ensure that the grown portion is null ptr
            unsafe {
                let ptr = self.0.as_mut_ptr().add(old_capacity);
                std::ptr::write_bytes(
                    ptr,
                    0,
                    std::mem::size_of::<*mut T>() * (new_capacity - old_capacity),
                );
            }
        }
    }
    pub fn push<A>(
        &mut self,
        mut item: A,
        boxing: fn(A) -> Box<T>,
        metadata: fn(&mut A) -> <T as Pointee>::Metadata,
        writing: fn(A, *mut T),
    ) {
        self.reserve_one();
        unsafe {
            let next_box_ptr: *mut T = *self.0.as_mut_ptr().add(self.0.len());
            if next_box_ptr.is_null() {
                // allocate new box
                self.0.push(Box::leak((boxing)(item)));
                return;
            }

            let next_ptr: &mut T = &mut *next_box_ptr;

            let old_metadata = std::ptr::metadata(next_ptr);
            let new_metadata = (metadata)(&mut item);
            if old_metadata == new_metadata {
                // reuse allocation
                (writing)(item, next_ptr);
                self.0.set_len(self.0.len() + 1);
            } else {
                // drop old and create new allocation
                std::alloc::dealloc(
                    next_ptr as *mut T as *mut u8,
                    std::alloc::Layout::for_value(next_ptr),
                );
                self.0.push(Box::leak((boxing)(item)));
            }
        }
    }
    pub fn clear(&mut self) {
        unsafe {
            for future in self.0.iter_mut() {
                // Drop the contents of the box, but not the box itself
                std::ptr::drop_in_place::<T>(*future);
            }
            self.0.set_len(0);
        }
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut().map(|ptr| unsafe { &mut **ptr })
    }
}
impl<T: ?Sized> Drop for ReusingBoxVec<T> {
    fn drop(&mut self) {
        self.clear();
        for i in 0..self.0.capacity() {
            unsafe {
                let next_box_ptr: *mut T = *self.0.as_mut_ptr().add(i);
                if next_box_ptr.is_null() {
                    break;
                }
                let next_ptr: &mut T = &mut *next_box_ptr;
                std::alloc::dealloc(
                    next_ptr as *mut T as *mut u8,
                    std::alloc::Layout::for_value(next_ptr),
                );
            }
        }
    }
}

pub struct ZipMany {
    futures: ReusingBoxVec<dyn FnMut(&mut DynRetainedValueContainer, &mut Context<'_>) -> Poll<()>>,
    retained_value_container: DynRetainedValueContainer,
}
impl ZipMany {
    pub fn push<'a, T: GPUFutureBlock<Returned = ()> + 'static>(&'a mut self, mut future: T) {
        // TODO: retrieve the next element, check its length, and see if we can
        let closure = move |container: &mut DynRetainedValueContainer, cx: &mut Context<'_>| {
            // This is ok because the future gets moved into a boxed closure and stays there until the closure gets dropped.
            let future_pinned = unsafe { Pin::new_unchecked(&mut future) };
            match future_pinned.poll(cx) {
                Poll::Ready(GPUFutureBlockReturnValue {
                    retained_values,
                    output: _output,
                }) => {
                    container.push(retained_values);
                    Poll::Ready(())
                }
                Poll::Pending => Poll::Pending,
            }
        };
        self.futures.push(
            closure,
            |closure| Box::new(closure),
            |closure| {
                std::ptr::metadata::<
                    dyn FnMut(&mut DynRetainedValueContainer, &mut Context<'_>) -> Poll<()>,
                >(closure as *mut _)
            },
            |mut closure, dst| unsafe {
                let closure_ref: &mut dyn FnMut(
                    &mut DynRetainedValueContainer,
                    &mut Context<'_>,
                ) -> Poll<()> = &mut closure;
                assert_eq!(std::ptr::metadata(dst), std::ptr::metadata(closure_ref));
                std::ptr::copy_nonoverlapping(
                    closure_ref as *mut _ as *mut u8,
                    dst as *mut u8,
                    std::mem::size_of_val(closure_ref),
                );
            },
        );
    }
    pub fn clear(&mut self) {
        self.futures.clear();
    }
}

impl Future for ZipMany {
    type Output = GPUFutureBlockReturnValue<(), DynRetainedValueContainer>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        for future in this.futures.iter_mut() {
            let result = (future)(&mut this.retained_value_container, cx);
            if result.is_pending() {
                assert!(this.retained_value_container.is_empty());
            }
        }
        if this.retained_value_container.is_empty() {
            Poll::Pending
        } else {
            Poll::Ready(GPUFutureBlockReturnValue {
                output: (),
                retained_values: std::mem::take(&mut this.retained_value_container),
            })
        }
    }
}
