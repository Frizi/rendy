/// A container type that updates the held data based on changes of external value.
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
    /// Create new tracked container
    pub fn new() -> Self {
        Self { tracked: None }
    }

    /// Provide tracking data to compare and update the value in case of change.
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

    /// Take current value out of the tracked container
    pub fn into_inner(self) -> Option<T> {
        self.tracked.map(|t| t.1)
    }
}
