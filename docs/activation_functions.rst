.. _activation_functions:

========================
Aktivasyon Fonksiyonları
========================

.. contents:: :local:

.. _activation_linear:

Lineer
======

Aktivasyonun girdiye (nörondan gelen ağırlıklı toplam) orantılı olduğu düz bir çizgi fonksiyonudur.

+-------------------------------------------------------+------------------------------------------------------+
| Fonksiyon                                             | Türev                                                |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z,m) = \begin{Bmatrix} z*m    \\               |       R'(z,m) = \begin{Bmatrix} m     \\             |
|                 \end{Bmatrix}                         |                   \end{Bmatrix}                      |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/linear.png                          | .. image:: images/linear_prime.png                   |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: linear                               |      :pyobject: linear_prime                         |
+-------------------------------------------------------+------------------------------------------------------+


.. rubric:: Artıları

- Bir aktivasyon aralığı verir, bu nedenle ikili (binary) bir aktivasyon değildir.
- Birkaç nöronu birbirine bağlayabiliriz ve 1'den fazlası ateşlenirse, max (veya softmax) alıp buna göre karar verebiliriz.

.. rubric:: Eksileri

- Bu fonksiyon için türev sabittir. Bu, gradient'in X ile bir ilişkisi olmadığı anlamına gelir.
- Sabit bir gradient'tir ve iniş (descent) sabit bir gradient üzerinde olacaktır.
- Tahminde bir hata varsa, back propagation tarafından yapılan değişiklikler sabittir ve girdideki değişime (delta(x)) bağlı değildir!



.. _activation_elu:

ELU
===

Exponential Linear Unit veya daha yaygın adıyla ELU, maliyeti daha hızlı sıfıra yakınsatan ve daha doğru sonuçlar üreten bir fonksiyondur. Diğer aktivasyon fonksiyonlarından farklı olarak, ELU pozitif bir sayı olması gereken ekstra bir alfa sabitine sahiptir.

ELU, negatif girdiler dışında RELU'ya çok benzer. Her ikisi de negatif olmayan girdiler için birim fonksiyon formundadır. Öte yandan, ELU çıktısı -α'ya eşit olana kadar yavaşça pürüzsüzleşirken, RELU keskin bir şekilde pürüzsüzleşir.

+-------------------------------------------------------+------------------------------------------------------+
| Fonksiyon                                             | Türev                                                |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z) = \begin{Bmatrix} z & z > 0 \\              |       R'(z) = \begin{Bmatrix} 1 & z>0 \\             |
|       α.( e^z – 1) & z <= 0 \end{Bmatrix}             |       α.e^z & z<0 \end{Bmatrix}                      |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/elu.png                             | .. image:: images/elu_prime.png                      |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: elu                                  |      :pyobject: elu_prime                            |
+-------------------------------------------------------+------------------------------------------------------+


.. rubric:: Artıları

- ELU, çıktısı -α'ya eşit olana kadar yavaşça pürüzsüzleşirken, RELU keskin bir şekilde pürüzsüzleşir.
- ELU, ReLU için güçlü bir alternatiftir.
- ReLU'dan farklı olarak, ELU negatif çıktılar üretebilir.

.. rubric:: Eksileri

- x > 0 için, [0, inf] çıktı aralığı ile aktivasyonu şişirebilir.


.. _activation_relu:

ReLU
====

Rectified Linear Units'in kısaltması olan yeni bir buluştur. Formülü aldatıcı bir şekilde basittir: :math:`max(0,z)`. Adına ve görünümüne rağmen, lineer değildir ve Sigmoid ile aynı faydaları (yani doğrusal olmayan fonksiyonları öğrenme yeteneği) daha iyi bir performansla sağlar.

+-------------------------------------------------------+------------------------------------------------------+
| Fonksiyon                                             | Türev                                                |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z) = \begin{Bmatrix} z & z > 0 \\              |       R'(z) = \begin{Bmatrix} 1 & z>0 \\             |
|       0 & z <= 0 \end{Bmatrix}                        |       0 & z<0 \end{Bmatrix}                          |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/relu.png                            | .. image:: images/relu_prime.png                     |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: relu                                 |      :pyobject: relu_prime                           |
+-------------------------------------------------------+------------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Artıları

- Kaybolan gradient (vanishing gradient) problemini önler ve düzeltir.
- ReLu, daha basit matematiksel işlemler içerdiği için tanh ve sigmoid'den daha az hesaplama maliyetlidir.

.. rubric:: Eksileri

- Sınırlamalarından biri, yalnızca bir sinir ağı modelinin gizli katmanlarında kullanılması gerektiğidir.
- Bazı gradient'ler eğitim sırasında kırılgan olabilir ve ölebilir. Bu, bir ağırlık güncellemesine neden olabilir ve bu da onun herhangi bir veri noktasında bir daha asla etkinleşmemesini sağlar. Başka bir deyişle, ReLu ölü nöronlara neden olabilir.
- Başka bir deyişle, ReLu'nun (x<0) bölgesindeki aktivasyonlar için, gradient 0 olacaktır, bu nedenle ağırlıklar iniş sırasında ayarlanmayacaktır. Bu, bu duruma giren nöronların hata/girdi varyasyonlarına yanıt vermeyi durduracağı anlamına gelir (çünkü gradient 0'dır, hiçbir şey değişmez). Buna ölen ReLu problemi denir.
- ReLu'nun aralığı :math:`[0, \infty)`'dur. Bu, aktivasyonu şişirebileceği anlamına gelir.

.. rubric:: Daha fazla bilgi

- `Deep Sparse Rectifier Neural Networks <http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf>`_ Glorot et al., (2011)
- `Yes You Should Understand Backprop <https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b>`_, Karpathy (2016)


.. _activation_leakyrelu:

LeakyReLU
=========

LeakyRelu, ReLU'nun bir çeşididir. :math:`z < 0` olduğunda 0 olmak yerine, sızdıran bir ReLU küçük, sıfır olmayan, sabit bir gradyan :math:`\alpha`'ya izin verir (Normalde, :math:`\alpha = 0.01`). Ancak, faydanın görevler arasındaki tutarlılığı şu anda belirsizdir. [1]_

+-------------------------------------------------------+------------------------------------------------------+
| Fonksiyon                                             | Türev                                                |
+-------------------------------------------------------+------------------------------------------------------+
| .. math::                                             | .. math::                                            |
|      R(z) = \begin{Bmatrix} z & z > 0 \\              |       R'(z) = \begin{Bmatrix} 1 & z>0 \\             |
|       \alpha z & z <= 0 \end{Bmatrix}                 |       \alpha & z<0 \end{Bmatrix}                     |
+-------------------------------------------------------+------------------------------------------------------+
| .. image:: images/leakyrelu.png                       | .. image:: images/leakyrelu_prime.png                |
|       :align: center                                  |      :align: center                                  |
|       :width: 256 px                                  |      :width: 256 px                                  |
|       :height: 256 px                                 |      :height: 256 px                                 |
+-------------------------------------------------------+------------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py   | .. literalinclude:: ../code/activation_functions.py  |
|       :pyobject: leakyrelu                            |      :pyobject: leakyrelu_prime                      |
+-------------------------------------------------------+------------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/



.. rubric:: Artıları

- Leaky ReLU'lar, küçük bir negatif eğime (0.01 veya civarı) sahip olarak "ölen ReLU" sorununu çözme girişimlerinden biridir.

.. rubric:: Eksileri

- Doğrusallığa sahip olduğu için karmaşık Sınıflandırma için kullanılamaz. Bazı kullanım durumları için Sigmoid ve Tanh'ın gerisinde kalır.

.. rubric:: Daha fazla bilgi

- `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/pdf/1502.01852.pdf>`_, Kaiming He et al. (2015)


.. _activation_sigmoid:

Sigmoid
=======

Sigmoid, girdi olarak gerçek bir değer alır ve 0 ile 1 arasında başka bir değer çıktılar. Çalışması kolaydır ve aktivasyon fonksiyonlarının tüm güzel özelliklerine sahiptir: doğrusal değildir, sürekli türevlenebilirdir, monotondur ve sabit bir çıktı aralığına sahiptir.

+-----------------------------------------------------+-----------------------------------------------------+
| Fonksiyon                                           | Türev                                               |
+-----------------------------------------------------+-----------------------------------------------------+
| .. math::                                           | .. math::                                           |
|      S(z) = \frac{1} {1 + e^{-z}}                   |      S'(z) = S(z) \cdot (1 - S(z))                  |
+-----------------------------------------------------+-----------------------------------------------------+
| .. image:: images/sigmoid.png                       | .. image:: images/sigmoid_prime.png                 |
|       :align: center                                |       :align: center                                |
|       :width: 256 px                                |       :width: 256 px                                |
+-----------------------------------------------------+-----------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py | .. literalinclude:: ../code/activation_functions.py |
|       :pyobject: sigmoid                            |       :pyobject: sigmoid_prime                      |
+-----------------------------------------------------+-----------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Artıları

- Doğası gereği doğrusal değildir. Bu fonksiyonun kombinasyonları da doğrusal değildir!
- Adım fonksiyonunun aksine analog bir aktivasyon verecektir.
- Pürüzsüz bir gradient'e de sahiptir.
- Bir sınıflandırıcı için iyidir.
- Aktivasyon fonksiyonunun çıktısı, lineer fonksiyonun (-inf, inf) aralığına kıyasla her zaman (0,1) aralığında olacaktır. Böylece aktivasyonlarımız bir aralıkta sınırlanmış olur. Güzel, o zaman aktivasyonları şişirmez.


.. rubric:: Eksileri

- Sigmoid fonksiyonunun her iki ucuna doğru, Y değerleri X'teki değişikliklere çok daha az yanıt verme eğilimindedir.
- "Kaybolan gradient" (vanishing gradients) sorununa yol açar.
- Çıktısı sıfır merkezli değildir. Bu, gradient güncellemelerinin farklı yönlerde çok uzağa gitmesine neden olur. 0 < çıktı < 1, ve bu optimizasyonu zorlaştırır.
- Sigmoid'ler doygunluğa ulaşır ve gradient'leri öldürür.
- Ağ daha fazla öğrenmeyi reddeder veya (kullanım durumuna ve gradient/hesaplama kayan nokta değeri sınırlarına ulaşana kadar) büyük ölçüde yavaşlar.

.. rubric:: Daha fazla bilgi

- `Yes You Should Understand Backprop <https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b>`_, Karpathy (2016)


.. _activation_tanh:

Tanh
====

Tanh, gerçek değerli bir sayıyı [-1, 1] aralığına sıkıştırır. Doğrusal değildir. Ancak Sigmoid'den farklı olarak, çıktısı sıfır merkezlidir.
Bu nedenle, pratikte tanh doğrusalsızlığı her zaman sigmoid doğrusalsızlığına tercih edilir. [1]_

+-----------------------------------------------------+-----------------------------------------------------+
| Fonksiyon                                           | Türev                                               |
+-----------------------------------------------------+-----------------------------------------------------+
| .. math::                                           | .. math::                                           |
|      tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}|      tanh'(z) = 1 - tanh(z)^{2}                     |
+-----------------------------------------------------+-----------------------------------------------------+
| .. image:: images/tanh.png                          | .. image:: images/tanh_prime.png                    |
|       :align: center                                |       :align: center                                |
|       :width: 256 px                                |       :width: 256 px                                |
+-----------------------------------------------------+-----------------------------------------------------+
| .. literalinclude:: ../code/activation_functions.py | .. literalinclude:: ../code/activation_functions.py |
|       :pyobject: tanh                               |       :pyobject: tanh_prime                         |
+-----------------------------------------------------+-----------------------------------------------------+

.. quick create tables with tablesgenerator.com/text_tables and import our premade template in figures/

.. rubric:: Artıları

- Tanh için gradient, sigmoid'den daha güçlüdür (türevler daha diktir).

.. rubric:: Eksileri

- Tanh'da da kaybolan gradient (vanishing gradient) sorunu vardır.


Softmax
=======

Softmax fonksiyonu, 'n' farklı olay üzerinden olayın olasılık dağılımını hesaplar. Genel bir deyişle, bu fonksiyon tüm olası hedef sınıfları üzerinden her bir hedef sınıfının olasılıklarını hesaplayacaktır. Daha sonra hesaplanan olasılıklar, verilen girdiler için hedef sınıfını belirlemede yardımcı olacaktır.


.. rubric:: Referanslar

.. [1] http://cs231n.github.io/neural-networks-1/
