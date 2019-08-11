class ClassDatasetTFRecord_v2(object):
    def __init__(self,im_width,im_height,im_channel,class_num,epoch,batch_size,is_shuffle=False,shuffle_buffer_size=None,augment_func=None):
            self.im_width    = im_width
            self.im_height   = im_height
            self.im_channel  = im_channel
            self.class_num   = class_num
            self.epoch       = epoch
            self.batch_size  = batch_size
            self.is_shuffle  = is_shuffle
            self.augment_func= augment_func
            self.dataset     = None
    
            if shuffle_buffer_size is None:
                self.shuffle_buffer_size = 4*batch_size
            else:
                self.shuffle_buffer_size = shuffle_buffer_size
            
        
        def __parse_proto(self,example_proto):
            features = tf.parse_single_example(example_proto,features=
                {
                'image':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.float32),
                'name' :tf.FixedLenFeature([],tf.string)
                })
    
            im_dim = tf.convert_to_tensor([self.im_channel,self.im_height,self.im_width])
            image = tf.transpose(tf.reshape(tf.decode_raw(features['image'],tf.uint8),im_dim),[1,2,0])
    
            if self.augment_func is not None:
                image = self.augment_func(image)
            
            label = tf.cast(tf.convert_to_tensor(features['label']),tf.int32)
            label = tf.one_hot(label,tf.convert_to_tensor(self.class_num), 1, 0) 
            image = tf.cast(image,tf.float32)
            label = tf.cast(label,tf.float32)
    
            return image,label

        def __call__(self,TFRecord_filename):

            if not isinstance(TFRecord_filename,list):
                TFRecord_filename = [TFRecord_filename]
            self.dataset = tf.data.TFRecordDataset(TFRecord_filename)
            self.dataset = self.dataset.map(self.__parse_proto,num_parallel_calls=8).repeat(self.epoch)
            if self.is_shuffle:
                self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)
            self.dataset = self.dataset.prefetch(self.batch_size).batch(self.batch_size)

            iterator = dataset.make_initializable_iterator()

            return iterator,self.dataset.output_types,self.dataset.output_shapes
