use std::mem::MaybeUninit;

pub struct RingBuffer<T, const N: usize> {
    items: [MaybeUninit<T>; N],
    start: u32,
    len: u32,
}

impl<T, const N: usize> Drop for RingBuffer<T, N> {
    fn drop(&mut self) {
        loop {
            if self.try_pop().is_none() {
                return;
            }
        }
    }
}

impl<T, const N: usize> RingBuffer<T, N> {
    pub const fn new() -> Self {
        Self {
            items: MaybeUninit::uninit_array(),
            start: 0,
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn push(&mut self, item: T) {
        if self.is_full() {
            panic!()
        };
        let mut location = self.start + self.len;
        if location >= N as u32 {
            location -= N as u32;
        }
        self.items[location as usize].write(item);
        self.len += 1;
    }
    pub fn pop(&mut self) -> T {
        self.try_pop().unwrap()
    }
    pub fn try_pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = unsafe { self.items[self.start as usize].assume_init_read() };
        self.len -= 1;
        self.start += 1;
        if self.start == N as u32 {
            self.start = 0;
        }
        Some(item)
    }
    pub fn pop_if_full(&mut self) -> Option<T> {
        if self.is_full() {
            Some(self.pop())
        } else {
            None
        }
    }
    pub fn is_full(&self) -> bool {
        self.len == N as u32
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn peek(&self) -> &T {
        if self.is_empty() {
            panic!()
        }
        unsafe { self.items[self.start as usize].assume_init_ref() }
    }
    pub fn peek_mut(&mut self) -> &mut T {
        if self.is_empty() {
            panic!()
        }
        unsafe { self.items[self.start as usize].assume_init_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::RingBuffer;

    #[test]
    fn test_len() {
        let mut buf: RingBuffer<usize, 3> = RingBuffer::new();
        buf.push(12);
        assert_eq!(buf.len(), 1);
        assert_eq!(*buf.peek(), 12);

        buf.push(23);
        assert_eq!(buf.len(), 2);
        assert_eq!(*buf.peek(), 12);

        buf.push(34);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.pop(), 12);
        assert_eq!(buf.pop(), 23);
        assert_eq!(buf.pop(), 34);
    }
}
