import random
import pickle as pkl
class RingBuffer:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.size = size
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
    
    def sample(self, sample_size):
        """randomly samples the given number of samples from the buffer.
            Returns a list containing specified number of data"""
        out = []
        while len(out)<sample_size:
            ind = int(random.random()*self.__len__())
            it = self.__getitem__(ind)
            out.append(it)

        return out

    def save(self, file_name):
        with open(file_name, 'wb+') as f:
            pkl.dump(self.data, f)
        

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            data = pkl.load(f)
        for d in data:
            self.append(d)
        
    def __getitem__(self, idx):
        if type(idx) == slice:
            data_len = len(self.data)

            if idx.start is None:
                start = self.start
            elif idx.start < 0:
                ## idx_start = idx.start + data_len - 1
                # print(idx.start,idx.stop,idx.step)
                # print("idx start: {} + self start: {} \% data_len: {} - 1".format(idx.start, self.start, data_len))
            #     if idx.start == -5 and self.start == 5 and data_len == 11:
            #         import pdb; pdb.set_trace()
                ##start = (idx_start + self.start) % data_len
                start = (idx.start + self.end) % data_len
            else:
                start = (idx.start + self.start) % data_len
            if idx.stop is None:
                stop = self.end # don't include the extra element (python slicing doesn't include upper bound, so we don't subtract one)
            elif idx.stop < 0:
                ## idx_stop = idx.stop + data_len - 1
                # print(idx.start,idx.stop,idx.step)
                # print("self.data: {}".format(self.data))
                # print("idx.stop: {} + self.end: {} \% data_len: {} - 1".format(idx.stop, self.end, data_len))
                
                ## stop = (idx_stop + self.start) % data_len
                stop = (idx.stop + self.end) % data_len
                # print("idx: {}".format(stop))
            else:
                stop = (idx.stop + self.start) % data_len
            if idx.step is None:
                step = 1
            else:
                step = idx.step

            if start > stop:
                return self.data[start::step] + self.data[:stop:step]
            else:
                return self.data[start:stop:step]
        if idx < 0: # again accounting for the extra entry in the buffer for negative indexing
            ## return self.data[(self.start + idx) % len(self.data)-1]
            return self.data[(self.end + idx) % len(self.data)]
        else:    
            return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        if self.start > self.end:
            return str(self.data[self.start:] + self.data[:self.end])
        else:
            return str(self.data[self.start:self.end])

if __name__ == '__main__':
    import random
    def test_slicing(buffer_len, n_inputs):
        rbf = RingBuffer(buffer_len)
        rbf_arr = []
        for i in range(n_inputs):
            sample = i # OR random.randint(-100,100)
            rbf.append(sample)
            if i >= n_inputs - buffer_len:
                rbf_arr.append(sample)
        print(rbf)
        print(rbf_arr)
        for i in range(len(rbf)):
            print("rbf[{}]: ring_buffer: {}, array: {}".format(i, rbf[i], rbf_arr[i]))
            assert rbf[i] == rbf_arr[i]
            print("rbf[{}:]: ring_buffer: {}, array: {}".format(i, rbf[i:], rbf_arr[i:]))
            assert rbf[i:] == rbf_arr[i:]
            print("rbf[:{}]: ring_buffer: {}, array: {}".format(i, rbf[:i], rbf_arr[:i]))
            assert rbf[:i] == rbf_arr[:i]
        for i in range(len(rbf)-1,0,-1):
            print("rbf[{}]: ring_buffer: {}, array: {}".format(-i, rbf[-i], rbf_arr[-i]))
            assert rbf[-i] == rbf_arr[-i]
            print("rbf[{}:]: ring_buffer: {}, array: {}".format(-i, rbf[-i:], rbf_arr[-i:]))
            assert rbf[-i:] == rbf_arr[-i:]
            print("rbf[:{}]: ring_buffer: {}, array: {}".format(-i, rbf[:-i], rbf_arr[:-i]))
            assert rbf[:-i] == rbf_arr[:-i]
        return rbf
    # test the ringbuffer slicing
    # print("Testing slicing with unfull buffer...")
    # buffer_len = 50
    # n_inputs = 7
    # test_slicing(buffer_len, n_inputs)

    print("Testing slicing with full buffer...")
    buffer_len = 12
    n_inputs = 12
    test_slicing(buffer_len, n_inputs)

    print("Testing slicing with overfull buffer...")
    buffer_len = 10
    n_inputs = 15
    test_slicing(buffer_len, n_inputs)

    print("Testing slicing with very overfull buffer...")
    buffer_len = 10
    n_inputs = 300
    rbf = test_slicing(buffer_len, n_inputs)

    # import pdb; pdb.set_trace()

    
    print("All tests passed!")