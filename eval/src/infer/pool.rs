//! Fixed pool of loaded scorers: bounds parallel inference to N (one ORT Session is `&mut self`,
//! so a session serializes; the pool restores N-way parallelism). Blocking, no async.
use std::sync::{Condvar, Mutex};

/// A fixed pool of `T` (each a loaded scorer). `acquire()` blocks until one is free.
pub struct Pool<T> {
    free: Mutex<Vec<T>>,
    cv: Condvar,
    size: usize,
}

/// RAII lease: derefs to `&T`; returns the item to the pool on drop.
pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> Pool<T> {
    pub fn new(items: Vec<T>) -> Pool<T> {
        let size = items.len();
        assert!(size > 0, "pool needs at least one item");
        Pool {
            free: Mutex::new(items),
            cv: Condvar::new(),
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Block until an item is free, then lease it. The lease returns it on drop.
    pub fn acquire(&self) -> PoolGuard<'_, T> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        while free.is_empty() {
            free = self.cv.wait(free).unwrap_or_else(|e| e.into_inner());
        }
        let item = free.pop();
        PoolGuard { pool: self, item }
    }
}

impl<T> std::ops::Deref for PoolGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item.as_ref().expect("guard holds an item until drop")
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let mut free = self.pool.free.lock().unwrap_or_else(|e| e.into_inner());
            free.push(item);
            self.pool.cv.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn pool_bounds_in_flight_to_n() {
        let pool = Arc::new(Pool::new(vec![(), ()])); // N=2
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max = Arc::new(AtomicUsize::new(0));
        let mut hs = vec![];
        for _ in 0..8 {
            let (p, f, m) = (pool.clone(), in_flight.clone(), max.clone());
            hs.push(std::thread::spawn(move || {
                let _g = p.acquire();
                let n = f.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(n, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(20));
                f.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        assert!(
            max.load(Ordering::SeqCst) <= 2,
            "max in-flight exceeded pool size"
        );
    }

    #[test]
    fn pool_reports_size() {
        let pool = Pool::new(vec![1u8, 2, 3]);
        assert_eq!(pool.size(), 3);
    }
}
