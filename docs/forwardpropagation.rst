.. _forwardpropagation:

==================
Forwardpropagation (İleri Yayılım)
==================

.. contents:: :local:

Simple Network (Basit Ağ)
==============

.. image:: images/neural_network_simple.png
    :align: center

Forward propagation (ileri yayılım) yapay sinir ağlarının (neural networks) prediction (tahmin) üretme sürecidir. Girdi (input) verisi ağ boyunca katman katman (layer by layer) ileri doğru aktarılır ve son katman prediction üretir. Yukarıdaki toy (oyuncak / basitleştirilmiş) ağ için tek bir forward pass matematiksel olarak şöyledir:

.. math::

  Prediction = A(\;A(\;X W_h\;)W_o\;)

:math:`A` burada :ref:`activation_relu` gibi bir activation function, :math:`X` girdi (input) ve :math:`W_h`, :math:`W_o` weight'lerdir.

Steps (Adımlar)
-----

1. Hidden layer'a giden weighted input (ağırlıklı giriş) için :math:`X` ile :math:`W_h` çarpılır.
2. Activation function uygulanır ve sonuç final (çıktı) katmana iletilir.
3. 2. adımı tekrarla; bu kez :math:`X` yerine hidden layer output'u :math:`H` kullanılır.


Code (Kod)
----

Tek hidden layer içeren bu basit ağda girdiyi ileri taşıyan (propagate) bir feed_forward() metodu yazalım. Bu metodun çıktısı modelin prediction değeridir.

.. literalinclude:: ../code/nn_simple.py
    :language: python
    :lines: 4-15

``x`` ağın input'u, ``Zo`` ve ``Zh`` weighted input (ağırlıklı giriş), ``Wo`` ve ``Wh`` ise weight matrisleridir.


Larger Network (Daha Büyük Ağ)
==============

Yukarıdaki basit ağ öğrenme amaçlı faydalıdır; ancak gerçek dünyada neural network'ler çok daha büyük ve karmaşıktır: daha fazla hidden layer, katman başına daha fazla neuron, input başına daha fazla feature (değişken), training set içinde daha çok örnek ve çoğu zaman çoklu output değişkenleri. Aşağıdaki biraz daha büyük ağ bize matrices (matrisler) ve büyük ağları eğitmekte kullanılan matrix operations (matris işlemleri) kavramını tanıtır.

.. image:: images/neural_network_w_matrices.png
    :align: center


Architecture (Mimari)
------------

Keyfi (arbitrarily) büyük input veya output'ları desteklemek için kodumuzu genişletilebilir (extensible) kılacak parametreler ekleriz: inputLayerSize, hiddenLayerSize, outputLayerSize. Hâlâ tek hidden layer kullanıyoruz; ancak artık farklı boyutlarda layer oluşturup farklı girdi/çıktı yapılarına uyum sağlayabiliyoruz.

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :lines: 6-8


Weight Initialization (Ağırlık Başlatma)
---------------------

Önceki basit örnekte ``Wh`` ve ``Wo`` scalar idi; şimdi weight değişkenleri numpy array (matris) olacak. Her array kendi layer'ına ait tüm weight'leri (her synapse için bir weight) tutar. Aşağıda numpy ``np.random.randn(rows, cols)`` fonksiyonu ile (ortalama 0, varyans 1 normal dağılımdan) weight matrislerini initialize ediyoruz.

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :pyobject: init_weights

``random.randn()`` kullanım örneği:

::

  arr = np.random.randn(1, 2)

  print(arr)
  >> [[-0.36094661 -1.30447338]]

  print(arr.shape)
  >> (1,2)

Bu weight matrislerinin boyutları için katı şartlar vardır: *rows* (satır) sayısı önceki layer'daki neuron sayısına, *columns* (sütun) sayısı sonraki layer'daki neuron sayısına eşit olmalıdır.

Random weight initialization üzerine iyi bir açıklama Stanford CS231 ders notlarında neural networks bölümünde bulunabilir [1]_.


Bias Terms (Bias Terimleri)
----------

:ref:`nn_bias` terimleri neuron activation output'larını sağa veya sola kaydırmamızı sağlar; bu, orijinden geçmeyen veri dağılımlarını modellemeye yardımcı olur.

Aşağıda numpy ``np.full()`` ile varsayılan değer ``0.2`` içeren iki adet 1‑boyutlu bias array oluşturuyoruz. İlk argüman boyut tuple'ı, ikinci argüman her hücre için default value.

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :pyobject: init_bias


Working with Matrices (Matrislerle Çalışma)
---------------------

Hızlı linear algebra teknikleri ve GPU avantajı için input, weight ve bias'ları matrislerde saklarız. Aşağıda ağ diyagramı matrix representation (temel matris gösterimi) ile tekrar verilmiştir.

.. image:: images/nn_with_matrices_displayed.png
    :align: center

Burada ne oluyor? Daha iyi anlamak için diyagramdaki her matrisi boyutlarına (dimensions) ve neden bu boyutlara sahip olduklarına odaklanarak inceleyelim. Boyutlar ağ mimarisi ve training set örnek sayısından doğal olarak türetilir.

.. rubric:: Matrix Dimensions (Matris Boyutları)

+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Var** | **Name**              | **Dimensions** | **Explanation**                                                                                                                                                                                                                                                                                                                                                                            |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``X``   | Input (Girdi)         | (3, 1)         | 3 satır (örnek) ve her satırda 1 attribute (özellik) (ör: height, price).                                                                                                                                                                                                                                                                                                                  |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Wh``  | Hidden weights        | (1, 2)         | Satır sayısı önceki layer attribute sayısı; sütun sayısı hidden layer neuron sayısı. İki layer arasındaki weight matrisi boyutu bağladığı layer boyutlarınca belirlenir. Her input→neuron bağlantısı için bir weight. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Bh``  | Hidden bias           | (1, 2)         | Hidden layer'daki her neuron için bir bias. ReLU uygulanmadan önce weighted input (Zh) üzerine eklenir. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Zh``  | Hidden weighted input | (1, 2)         | X · Wh dot product'ı ile elde edilir; ardından Bh eklenir. Boyutlar matris çarpım kurallarından gelir. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``H``   | Hidden activations    | (3, 2)         | Zh üzerine ReLU uygulanır. Satırlar örnek sayısı (3), sütunlar neuron sayısı (2). Her sütun ilgili neuron aktivasyonlarını içerir. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Wo``  | Output weights        | (2, 2)         | Satırlar hidden layer neuron sayısı, sütunlar output layer neuron sayısı. Her bağlantı için bir weight. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Bo``  | Output bias           | (1, 2)         | Output layer'daki her neuron için bir bias değeri (sütun). |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``Zo``  | Output weighted input | (3, 2)         | H · Wo + Bo ile hesaplanır. Satırlar örnek, sütunlar output neuron. |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``O``   | Output activations    | (3, 2)         | Her satır bir örnek için prediction; her sütun farklı bir output attribute (ör: satış ve adet, boy ve kilo). |
+---------+-----------------------+----------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Dynamic Resizing (Dinamik Boyutlanma)
----------------

Devam etmeden önce network architecture veya training set boyutu değiştiğinde matris boyutlarının nasıl değiştiğine bakalım. Örneğin 2 input neuron, 3 hidden neuron, 2 output neuron ve 4 observation içeren bir ağ kuralım.

.. image:: images/dynamic_resizing_neural_network_4_obs.png
    :align: center

Aynı layer / neuron sayılarını koruyup observation sayısını **1**'e düşürelim:

.. image:: images/dynamic_resizing_neural_network_1_obs.png
    :align: center

Görüldüğü üzere sütun sayıları sabit kalır; değişen sadece satır sayısıdır ve o da training set boyutuna bağlıdır. Weight matrix boyutları değişmez. Böylece aynı kod ile değişken sayıda observation işleyebiliriz.


Refactoring Our Code (Kodu Yeniden Düzenleme)
---------------------

Artık scalar yerine matrix kabul eden yeni feed forward kodu:

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :pyobject: feed_forward

.. rubric:: Weighted Input (Ağırlıklı Giriş)

İlk değişiklik weighted input hesabının matrislere uyarlanmasıdır. Dot product ile input matrix, bir sonraki layer neuron'larına giden weight matrisiyle çarpılır. Sonra bias vector eklenir (matrix addition).

::

  Zh = np.dot(X, Wh) + Bh

.. image:: images/neural_network_matrix_weighted_input.png
    :align: center

``Bh``'nin ilk değeri X·Wh sonucunun ilk sütunundaki tüm satırlara, ikinci değeri ikinci sütundaki tüm elemanlara eklenir. Sonuç ``Zh`` matrisidir; her sütun bir hidden neuron, her satır bir observation. Ağ *fully-connected* olduğundan her neuron→neuron bağlantısı için bir weight vardır.

Aynı süreç output layer için tekrarlanır; input artık ``H`` ve weights ``Wo``.

.. rubric:: ReLU Activation

İkinci değişiklik ReLU'nun matris üzerinde elementwise uygulanmasıdır. Küçük ama gerekli bir adım. ``np.maximum()`` hem scalar hem array ile çalışabildiği için uygundur.

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :pyobject: relu

Hidden layer activation adımında ``np.maximum(0, Z)`` ile tüm negatif değerleri 0 yaparız. Aynısı output layer için ``Zo`` üzerinde tekrarlanır.


Final Result (Nihai Sonuç)
------------

Tüm parçaları birleştirince matrislerle forward propagation kodu aşağıdaki gibidir.

.. literalinclude:: ../code/nn_matrix.py
    :language: python
    :lines: 6-60


.. rubric:: References

.. [1] http://cs231n.github.io/neural-networks-2/#init

