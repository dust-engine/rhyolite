use std::future::Future;

pub fn unwrap_future<F: Future>(future: F) -> F::Output {
    let future = std::pin::pin!(future);
    let result = future.poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
    match result {
        std::task::Poll::Ready(val) => val,
        std::task::Poll::Pending => panic!("Future did not resolve"),
    }
}
