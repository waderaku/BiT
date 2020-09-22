import tensorflow as tf


class BigTransfer(tf.keras.Model):

    def __init__(self, num_classes, module):
        super().__init__()

        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(
            num_classes, kernel_initializer='zeros')
        self.bit_model = module

    # predict関数を最大の選択肢だのindex出力するようにしてる。後でテストするときに視覚的にわかりやすいためだけどこの辺は仕様次第かな
    def predict(self, image):
        return tf.argmax(self(image)[0]).numpy()

    # tf,functionつけてみたけどあんま早くないかも。Bitモデル自体がtensorflowオブジェクトの計算に最適化されてなさそう。
    @tf.function
    def call(self, images):
        bit_embedding = self.bit_model(images)
        # 論文のコードだとsoftmaxかけられてなかったけどさすがにちがくね？ってことでsofmaxかけてる
        # あんま精度でなかったら外してもいいかも
        return tf.keras.activations.softmax(self.head(bit_embedding))
