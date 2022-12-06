#![feature(min_specialization)]
#![feature(array_methods)]
#![feature(waker_getters)]
#![feature(pin_macro)]
#![feature(generators, generator_trait, generator_clone)]
#![feature(return_position_impl_trait_in_trait)]
#![feature(async_fn_in_trait)]
#![feature(iter_from_generator)]
#![feature(async_closure)]
#![feature(trait_alias)]
#![feature(type_alias_impl_trait)]
use std::{task::Poll, pin::Pin, ops::Generator};

use ash::vk;
use async_ash::future::*;

struct Item {

}
impl Drop for Item {
    fn drop(&mut self) {
        println!("Dropping Item");
    }
}

fn main() {
    use async_ash_macro::{join, commands};
    let mut aaa = Item{};
    let mut bbb = Item{};
    let block = commands! {
        //aa.push(1);

        let block3 = CopyBufferFuture { str: "special" };
        let block1 = commands! {
            let mut aa = import!(aaa);
            let sss: String = "Testtt".to_string();
            let fut1 = CopyBufferFuture{ str: sss.as_str()};
            fut1.await;
            drop(sss);
            let fut1 = CopyBufferFuture{ str: "prev2"};
            fut1.await;
            1_u32
        };
    
        let block2 = commands! {
            let mut bb = import!(bbb);
            let fut1 = CopyBufferFuture{ str: "A"};
            fut1.await;
            let fut1 = CopyBufferFuture{ str: "B"};
            fut1.await;
            2_u64
        }; 

        let (a, b, c) = join!(block1, block2, block3).await;
        println!("a: {:?}, b: {:?}, c: {:?}", a, b, c);
    };

    block.record_all(vk::CommandBuffer::null());
}
fn test(a: &mut vk::CommandBuffer) {

}
struct CopyBufferFuture<'a> {
    str: &'a str,
}
impl<'a> GPUCommandFuture for CopyBufferFuture<'a> {
    type Output = ();
    #[inline]
    fn record(self: Pin<&mut Self>, command_buffer: vk::CommandBuffer) -> Poll<Self::Output> {
        println!("{}", self.str);
        Poll::Ready(())
    }
    fn context(self: Pin<&mut Self>, ctx: &mut GPUCommandFutureContext) {
    }
}


/*
on the host, build the buffer with all the VkAccelerationStructureInstanceKHR to be instanceBuffer.
device_instance_buffer.copy(instanceBuffer);

let tlas = tlas_build(&instanceBuffer, &mut scratchData).await;

let sbt = device_sbt.copy(sbt_buffer);

trace_rays(&tlas, &sbt, &mut targetImage)
*/