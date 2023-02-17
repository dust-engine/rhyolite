use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;

pub fn use_state<T>(
    this: &mut Option<T>,
    init: impl FnOnce() -> T,
    update: impl FnOnce(&mut T),
) -> &mut T {
    if let Some(inner) = this {
        update(inner)
    }
    this.get_or_insert_with(init)
    // TASK fors tomorrow: use_perframe_state
}

pub fn use_cached_state<T>(
    this: &mut Option<Arc<T>>,
    create: impl FnOnce() -> T,
    should_update: impl FnOnce(&T) -> bool,
) -> Arc<T> {
    if let Some(inner) = this {
        if should_update(inner) {
            *inner = Arc::new(create());
        }
        inner.clone()
    } else {
        let item = Arc::new(create());
        *this = Some(item.clone());
        item
    }
}

use std::sync::mpsc;
// probably needs a mpsc channel.
pub struct PerFrameState<T> {
    receiver: mpsc::Receiver<T>,
    sender: mpsc::Sender<T>,
    pending_items: usize,
}
// Safety: We do not expose &self.receiver or &self.sender to the outside.
unsafe impl<T> Sync for PerFrameState<T> {}
impl<T> Default for PerFrameState<T> {
    fn default() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            receiver,
            sender,
            pending_items: 0,
        }
    }
}
pub struct PerFrameContainer<T> {
    sender: mpsc::Sender<T>,
    item: Option<T>,
}
unsafe impl<T> Sync for PerFrameContainer<T> {}
impl<T> Deref for PerFrameContainer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().unwrap()
    }
}
impl<T> DerefMut for PerFrameContainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().unwrap()
    }
}
impl<T> Drop for PerFrameContainer<T> {
    fn drop(&mut self) {
        self.sender.send(self.item.take().unwrap()).unwrap();
    }
}
pub fn use_per_frame_state<T>(
    this: &mut PerFrameState<T>,
    create: impl FnOnce() -> T,
    reuse: impl FnOnce(&mut T),
) -> PerFrameContainer<T> {
    let item = this
        .receiver
        .try_recv()
        .map(|mut item| {
            reuse(&mut item);
            item
        })
        .unwrap_or_else(|err| match err {
            mpsc::TryRecvError::Empty => {
                this.pending_items += 1;
                create()
            }
            mpsc::TryRecvError::Disconnected => panic!(),
        });
    PerFrameContainer {
        sender: this.sender.clone(),
        item: Some(item),
    }
}
pub fn use_per_frame_state_blocking<T>(
    this: &mut PerFrameState<T>,
    max_pending: usize,
    create: impl FnOnce() -> T,
    reuse: impl FnOnce(&mut T),
) -> PerFrameContainer<T> {
    let item = if this.pending_items < max_pending {
        this.receiver
            .try_recv()
            .map(|mut item| {
                reuse(&mut item);
                item
            })
            .unwrap_or_else(|err| match err {
                mpsc::TryRecvError::Empty => {
                    this.pending_items += 1;
                    create()
                }
                mpsc::TryRecvError::Disconnected => panic!(),
            })
    } else {
        this.receiver
            .recv()
            .map(|mut item| {
                reuse(&mut item);
                item
            })
            .unwrap_or_else(|_| panic!())
    };
    PerFrameContainer {
        sender: this.sender.clone(),
        item: Some(item),
    }
}
