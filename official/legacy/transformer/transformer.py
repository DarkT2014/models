# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

import tensorflow as tf

from official.legacy.transformer import attention_layer
from official.legacy.transformer import embedding_layer
from official.legacy.transformer import ffn_layer
from official.legacy.transformer import metrics
from official.legacy.transformer import model_utils
from official.legacy.transformer.utils.tokenizer import EOS_ID
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.ops import beam_search

# Disable the not-callable lint error, since it claims many objects are not
# callable when they actually are.
# pylint: disable=not-callable


def create_model(params, is_train):
  """Creates transformer model."""
  '''
  Q: 为什么要使用这个函数? 我理解直接调用Transformer的init函数就可以了啊?
  而且不同模型对应的逻辑应该也是不同的, 因此在这里即使抽象出来一个共用的接口, 在调用时, 调用语句也会不可避免的处于不同的逻辑之中啊。


  A: 
  当你在训练模型时, 你需要将模型的输出与真实的目标进行比较, 并计算损失。
  这就需要你创建一个模型, 它接受输入和目标, 输出预测结果, 并计算损失。
  这就是create_model在is_train=True时所创建的模型。

  相反, 当你在推理(预测)时, 你并不需要计算损失, 你只需要模型的预测结果。
  因此, 你需要一个不同的模型, 它只接受输入, 输出预测结果。
  这就是create_model在is_train=False时所创建的模型。

  将这个逻辑抽象到一个函数中的好处是, 你可以在代码中多次重用这个函数, 而不需要在每次需要创建模型时都写出完整的逻辑。
  此外, 如果你在未来需要修改模型的创建逻辑, 你只需要修改这个函数, 而不需要找出并修改代码中的所有相关部分。

  

  至于你提到的问题, 即不同模型的调用逻辑可能不同, 这是对的。
  但这并不妨碍我们在创建模型时使用相同的接口。

  实际上, 这正是Keras(和其他深度学习框架)的设计理念之一:
    尽管不同的模型可能有着完全不同的内部逻辑, 但它们都可以通过相同的接口(比如fit、predict等方法)进行训练和推理。
  这使得我们可以编写通用的训练和推理代码, 而不需要关心模型的具体实现。
  '''



  '''
  在TensorFlow(尤其是TensorFlow 2.0和Keras API)中, 模型的构建通常分为以下几个步骤:

    1 定义输入张量。在这个例子中, 输入张量是inputs和targets, 它们的形状分别为(None,), 表示它们可以是任意长度的序列。

    2 创建模型的主体部分, 即神经网络的各个层。在这个例子中, 这个部分是Transformer类的实例。

    3 将输入张量传递给模型的主体部分, 得到输出张量。在这个例子中, 这个操作是logits = internal_model([inputs, targets], training=is_train)。

    4 如果有需要, 对输出张量进行进一步处理, 比如添加损失函数、计算评估指标等。在这个例子中, 训练逻辑的部分计算了损失函数并添加到了模型上。

    5 使用tf.keras.Model类将输入张量和输出张量封装成一个完整的模型。在这个例子中, 这个操作是model = tf.keras.Model([inputs, targets], logits)。

  创建好模型之后, 你可以像使用其他Keras模型一样使用它。例如, 你可以调用model.compile()来配置训练过程, 然后调用model.fit()来训练模型。
  同样, 在推理阶段, 你可以调用model.predict()来预测新的序列。
  '''

  with tf.name_scope("model"):
    if is_train:#训练逻辑

      # 1 定义输入张量。
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      # TensorFlow中, 将张量的某个维度设置为None意味着该维度可以是任意大小
      # 这两个输入都是整数张量, 它们的形状为(None,), 表示它们可以是任意长度的序列
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")


      # 2 创建模型的主体部分, 即神经网络的各个层。
      internal_model = Transformer(params, name="transformer_v2")

      #3 将输入张量传递给模型的主体部分, 得到输出张量。
      logits = internal_model([inputs, targets], training=is_train)
      # 这一行实际上是在调用 Transformer 对象的 call 方法，将输入数据 inputs 和 targets 传递给模型，并执行前向传播操作。training 参数指示模型当前是否处于训练模式。

      # 在 TensorFlow 中，模型对象（如 internal_model）可以被当作函数来调用。这是因为模型对象重载了 __call__ 特殊方法。
      # 当你像这样调用模型对象时，实际上是在调用模型对象的 call 方法。
      # 所以，internal_model([inputs, targets], training=is_train) 这一行实际上是在执行模型的前向传播操作
      # ，而不是在初始化一个新的模型。


      vocab_size = params["vocab_size"]


      label_smoothing = params["label_smoothing"]
      #   Label smoothing(标签平滑)是一种正则化技术，用于改善分类模型在训练集上的性能和泛化能力。
      # 它通过调整分类任务中的目标标签，将原始的"one-hot"标签分布调整为更平滑的分布。
      #   通常情况下，分类任务中的目标标签是使用"one-hot"编码表示的，即只有一个类别的标签为1，其他类别的标签都为0。
      # 然而，这种表示方式会使得模型过于自信地预测一个类别，而忽略其他可能性。
      #   Label smoothing 通过将目标标签中的1替换为一个较小的正数(如0.9)，并将其他标签（0）替换为一个较小的负数（如0.1/（类别数-1）），从而平滑了标签的分布。
      #   使用标签平滑的目的是减少模型对训练集中特定样本的过拟合，提高模型的泛化能力。它可以帮助模型更加鲁棒地处理不确定性，并降低过度自信的预测。

      if params["enable_metrics_in_training"]:
        logits = metrics.MetricLayer(vocab_size)([logits, targets])
        # MetricLayer 是一个自定义的 Keras 层，它的作用是在训练过程中计算一些额外的度量指标。
        # 它接受模型的预测结果和实际的目标作为输入，然后计算这两者之间的某些度量指标（例如准确率、召回率等）。
        # 这个层的输出仍然是模型的预测结果（即 logits），但它会将计算出的度量指标添加到模型的 metrics 列表中，以便在训练过程中进行监控。

        # 这段代码对模型的预测结果和训练过程的影响可能会因情况而异。
        # 在大多数情况下，它不会改变模型的预测结果，因为 MetricLayer 不会改变其输入的 logits。
        # 然而，它可能会影响训练过程，因为它会将计算出的度量指标添加到模型的 metrics 列表中，这可能会改变训练过程的行为。
        # 例如，如果你在 model.compile() 中指定了一个基于度量指标的早停回调或学习率调度器，那么这个层就可能会影响训练过程。



      # 这行代码的作用是在 logits 上添加一个 Lambda 层。
      # 这个 Lambda 层的作用是对其输入 logits 执行一个简单的函数，这个函数就是 lambda x: x，它的作用是返回其输入。
      # 也就是说，这个 Lambda 层实际上并没有改变 logits。
      # 这个层的存在主要是为了给 logits 添加一个名字（"logits"）和数据类型（tf.float32）。这可能是为了让模型的输出更容易被理解和使用。
      logits = tf.keras.layers.Lambda(
          lambda x: x, name="logits", dtype=tf.float32)(
              logits)
      # logits : 模型的输出
      
      
      # 5 使用tf.keras.Model类将输入张量和输出张量封装成一个完整的模型。
      model = tf.keras.Model([inputs, targets], logits)

      # 计算loss, 即根据logits与targets的差距计算loss
      loss = metrics.transformer_loss(logits, targets, label_smoothing,
                                      vocab_size)
      

      model.add_loss(loss)
      #  这一行代码将计算出的损失添加到了模型的损失列表中。
      # 在训练过程中，Keras 会自动计算这个列表中的所有损失，并将它们相加得到总损失。
      # 然后，Keras 会使用反向传播算法优化这个总损失。

      # 所以，这段代码并没有直接完成反向传播，它只是设置了模型的输入、输出和损失函数。
      # 实际的反向传播操作是在模型训练过程中由 Keras 自动完成的。
      # 你可以通过调用 model.fit() 方法来开始训练模型，Keras 会在每个训练步骤中自动计算损失，执行反向传播并更新模型参数。
      return model

    else:#预测逻辑
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      internal_model = Transformer(params, name="transformer_v2")
      ret = internal_model([inputs], training=is_train)
      outputs, scores = ret["outputs"], ret["scores"]
      return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
  """Transformer model with Keras.

  Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

  The Transformer model consists of an encoder and decoder. The input is an int
  sequence (or a batch of sequences). The encoder produces a continuous
  representation, and the decoder uses the encoder output to generate
  probabilities for the output sequence.
  """

  def __init__(self, params, name=None):
    """Initialize layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(Transformer, self).__init__(name=name)
    self.params = params
    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        params["vocab_size"], params["hidden_size"])
    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.params["hidden_size"])

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item (optional), targets: None or int tensor with shape
          [batch_size, target_length].
      training: boolean, whether in training mode or not.

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: int tensor with shape [batch_size, decoded_length]
          scores: float tensor with shape [batch_size]}
      Even when float16 is used, the output tensor(s) are always float32.

    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    inputs = inputs if isinstance(inputs, list) else [inputs]
    if len(inputs) == 2:
      inputs, targets = inputs[0], inputs[1]
    else:
      # Decoding path.
      inputs, targets = inputs[0], None
      if self.params["padded_decode"]:
        if not self.params["num_replicas"]:
          raise NotImplementedError(
              "Padded decoding on CPU/GPUs is not supported.")
        decode_batch_size = int(self.params["decode_batch_size"] /
                                self.params["num_replicas"])
        inputs.set_shape([decode_batch_size, self.params["decode_max_length"]])

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = model_utils.get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias, training)
      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias, training)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias, training)
        return logits

  def encode(self, inputs, attention_bias, training):
    """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.embedding_softmax_layer(inputs)
      embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
      inputs_padding = model_utils.get_padding(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])

      with tf.name_scope("add_pos_encoding"):
        pos_encoding = self.position_embedding(inputs=embedded_inputs)
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        encoder_inputs = embedded_inputs + pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params["layer_postprocess_dropout"])

      return self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)

  def decode(self, targets, encoder_outputs, attention_bias, training):
    """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
    """
    with tf.name_scope("decode"):
      # Prepare inputs to decoder layers by shifting targets, adding positional
      # encoding and applying dropout.
      with tf.name_scope("shift_targets"):
        # Shift targets to the right, and remove the last element
        targets = tf.pad(targets, [[0, 0], [1, 0]])[:, :-1]
      decoder_inputs = self.embedding_softmax_layer(targets)
      decoder_inputs = tf.cast(decoder_inputs, self.params["dtype"])
      attention_bias = tf.cast(attention_bias, self.params["dtype"])
      with tf.name_scope("add_pos_encoding"):
        length = tf.shape(decoder_inputs)[1]
        pos_encoding = self.position_embedding(decoder_inputs)
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        decoder_inputs += pos_encoding
      if training:
        decoder_inputs = tf.nn.dropout(
            decoder_inputs, rate=self.params["layer_postprocess_dropout"])

      # Run values
      decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
          length, dtype=self.params["dtype"])
      outputs = self.decoder_stack(
          decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          attention_bias,
          training=training)
      logits = self.embedding_softmax_layer(outputs, mode="linear")
      logits = tf.cast(logits, tf.float32)
      return logits

  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self.position_embedding(
        inputs=None, length=max_decode_length + 1)
    timing_signal = tf.cast(timing_signal, self.params["dtype"])
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length, dtype=self.params["dtype"])

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += timing_signal[i]
      if self.params["padded_decode"]:
        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_bias,
          cache.get("encoder_decoder_attention_bias"),
          training=training,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)
      logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
    if self.params["padded_decode"]:
      batch_size = encoder_outputs.shape.as_list()[0]
      input_length = encoder_outputs.shape.as_list()[1]
    else:
      batch_size = tf.shape(encoder_outputs)[0]
      input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length, training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = (
        max_decode_length if self.params["padded_decode"] else 0)
    num_heads = self.params["num_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    cache = {
        "layer_%d" % layer: {
            "k":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
                    dtype=self.params["dtype"]),
            "v":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
                    dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=EOS_ID,
        padded_decode=self.params["padded_decode"],
        dtype=self.params["dtype"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}


class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer, params):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, x, *args, **kwargs):
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output
    y = self.layer(y, *args, **kwargs)

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    return x + y


class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, encoder_inputs, attention_bias, inputs_padding, training):
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
  """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

  def __init__(self, params):
    super(DecoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the decoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      enc_dec_attention_layer = attention_layer.Attention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(enc_dec_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(DecoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self,
           decoder_inputs,
           encoder_outputs,
           decoder_self_attention_bias,
           attention_bias,
           training,
           cache=None,
           decode_loop_step=None):
    """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: A tensor with shape [batch_size, target_length,
        hidden_size].
      encoder_outputs: A tensor with shape [batch_size, input_length,
        hidden_size]
      decoder_self_attention_bias: A tensor with shape [1, 1, target_len,
        target_length], the bias for decoder self-attention layer.
      attention_bias: A tensor with shape [batch_size, 1, 1, input_length], the
        bias for encoder-decoder attention layer.
      training: A bool, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": A tensor with shape [batch_size, i, key_channels],
                     "v": A tensor with shape [batch_size, i, value_channels]},
                       ...}
      decode_loop_step: An integer, the step number of the decoding loop. Used
        only for autoregressive inference on TPU.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      enc_dec_attention_layer = layer[1]
      feed_forward_network = layer[2]

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.name_scope(layer_name):
        with tf.name_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs,
              decoder_self_attention_bias,
              training=training,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
        with tf.name_scope("encdec_attention"):
          decoder_inputs = enc_dec_attention_layer(
              decoder_inputs,
              encoder_outputs,
              attention_bias,
              training=training)
        with tf.name_scope("ffn"):
          decoder_inputs = feed_forward_network(
              decoder_inputs, training=training)

    return self.output_normalization(decoder_inputs)
