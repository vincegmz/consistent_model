from torch.utils.data.sampler import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from collections import defaultdict
import random
from torch.utils.data import Sampler

class ClassBasedBatchSampler(Sampler):
    def __init__(self,labels,class_scores,num_samples,batch_size):
        self.labels = labels
        self.class_scores = class_scores
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.index_by_class = defaultdict(list)
        for id,label in enumerate(labels):
            self.index_by_class[label].append(id)

    def __iter__(self) -> Iterator:
        num_batches = self.num_samples//self.batch_size
        for i in range(num_batches):
            chosen_class = random.choices(list(self.class_scores.keys()),weights=list(self.class_scores.values()),k=1)[0]
            if len(self.index_by_class) >= self.batch_size:
                yield random.sample(self.index_by_class[chosen_class],self.batch_size)

    
