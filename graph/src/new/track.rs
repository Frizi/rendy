#[derive(Debug)]
pub struct Track<K: PartialEq, T> {
    tracked: Option<(K, T)>,
}

impl<K: PartialEq, T> Default for Track<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: PartialEq, T> Track<K, T> {
    pub fn new() -> Self {
        Self { tracked: None }
    }

    pub fn track(&mut self, key: K, create: impl FnOnce(Option<T>) -> T) -> &T {
        if let Some((saved, trackable)) = self.tracked.take() {
            if saved == key {
                self.tracked.replace((saved, trackable));
            } else {
                self.tracked.replace((key, create(Some(trackable))));
            }
        } else {
            self.tracked.replace((key, create(None)));
        }
        self.tracked.as_ref().map(|t| &t.1).unwrap()
    }

    pub fn take(&mut self) -> Option<T> {
        self.tracked.take().map(|t| t.1)
    }
}
