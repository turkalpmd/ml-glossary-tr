.. _architectures:

===========
Mimariler
===========

.. contents:: :local:

Autoencoder
===========

Bir autoencoder, girdisini çıktısına kopyalamaya çalışan bir feedforward (ileri beslemeli) yapay sinir ağı türüdür. İç yapısında girdiyi temsil etmek için kullanılan **kod**u (code) tanımlayan **h** isimli gizli bir katman bulunur. Ağ iki kısımdan oluşur:

* *Encoder* (kodlayıcı) fonksiyon: :math:`h = f(x)`.
* *Decoder* (çözücü) fonksiyon: yeniden yapı (rekonstrüksiyon) üretir: :math:`r = g(h)`.

Aşağıdaki şekil bu mimariyi göstermektedir.

.. figure:: images/autoencoder_architecture.png
       :align: center
       :width: 200 px

       Source [#autoenc]_

Autoencoder girdi verisini daha düşük boyutlu bir koda sıkıştırır ve sonra bu temsilden çıktıyı yeniden inşa eder. Kod (code), girdinin kompakt bir "özeti" ya da "sıkıştırması"dır; aynı zamanda *gizil uzay (latent space) temsili* olarak da adlandırılır.

Eğer bir autoencoder her yerde :math:`g(f(x))=x` öğrenmiş olsaydı pek faydalı olmazdı; bunun yerine autoencoder'lar girdi kopyasını kusursuz çıkartamayacak şekilde tasarlanır. Böylece yalnızca yaklaşık kopyalayabilir ve sadece eğitim verisine benzeyen girdileri iyi kopyalayabilir. Model, girdinin hangi yönlerini koruyacağını önceliklendirmeye zorlandığından verinin faydalı özelliklerini öğrenir.

Bir autoencoder inşa etmek için üç şeye ihtiyaç vardır: bir kodlama yöntemi (encoding), bir dekodlama yöntemi (decoding) ve çıktı ile hedefi karşılaştıracak bir kayıp (loss) fonksiyonu.

Hem encoder hem decoder tam bağlantılı (fully-connected) feedforward sinir ağlarıdır. Kod, seçtiğimiz boyutta tek katmanlı bir yapay sinir ağı katmanıdır. Kod katmanındaki düğüm (nöron) sayısı (*code size*) eğitimden önce belirlenen bir *hiperparametre*dir.

Aşağıdaki şekil autoencoder mimarisini gösterir. Önce girdi, kodu üretmek için tam bağlantılı bir sinir ağı olan encoder'dan geçer. Benzer yapıda olan decoder yalnızca kodu kullanarak çıktıyı üretir. Amaç girdiye özdeş bir çıktı elde etmektir.

.. figure:: images/autoencoder_2.png
      :align: center
      :width: 500 px

      Source [#2a]_

Geleneksel olarak autoencoder'lar boyut indirgeme (dimensionality reduction) veya özellik öğrenme (feature learning) için kullanıldı. Daha yakın zamanda autoencoder'lar ile gizil değişken (latent variable) modelleri arasındaki teorik bağlantılar, autoencoder'ları üretici (generative) modellemenin ön saflarına taşıdı. Bir sıkıştırma yöntemi olarak alternatiflerinden daha iyi performans göstermezler ve veri-özel olmaları onları genel amaçlı bir teknik olarak pratik olmaktan uzaklaştırır.

Genel olarak üç yaygın kullanım senaryosu vardır:

* **Veri gürültü giderme (data denoising):** Denoising autoencoder'ların bir görüntüyü otomatik olarak gürültüsüzleştirmek için değil, gizli katmanların daha dayanıklı filtreler öğrenmesine ve aşırı uyum (overfitting) riskini azaltmaya yardım etmek için icat edildiği unutulmamalıdır.
* **Boyut indirgeme:** Yüksek boyutlu veriyi görselleştirmek zordur. t-SNE [#tsne]_ en yaygın kullanılan yöntemdir ancak çok yüksek boyutlarla (genelde 32 üstü) zorlanır. Bu nedenle autoencoder'lar ön işleme adımı olarak boyutu düşürmekte kullanılabilir; sıkıştırılmış temsil t-SNE tarafından veriyi 2B alanda görselleştirmek için kullanılır.
* **Varyasyonel Autoencoder (VAE):** Autoencoder'ların daha modern ve karmaşık bir kullanım alanıdır. VAE, klasik (vanilla) autoencoder'larda olduğu gibi keyfi bir fonksiyon öğrenmek yerine girdi verisini modelleyen olasılık dağılımının parametrelerini öğrenir. Bu dağılımdan noktalar örnekleyerek VAE'yi üretici bir model olarak da kullanabiliriz [#vae]_.


.. rubric:: Model

PyTorch ile örnek bir implementasyon.

.. literalinclude:: ../code/autoencoder.py
      :pyobject: Autoencoder

.. rubric:: Eğitim

.. literalinclude:: ../code/autoencoder.py
      :pyobject: train

.. rubric:: Ek okuma

- `Konvolüsyonel Autoencoder'lar <https://pgaleone.eu/neural-networks/2016/11/24/convolutional-autoencoders/>`_
- `Deep Learning Kitabı <http://www.deeplearningbook.org/contents/autoencoders.html>`_


CNN
===

Konvolüsyonel sinir ağı (*convolutional neural network* - *CNN*), en az bir konvolüsyon katmanı içeren bir feed-forward sinir ağıdır. Bu tür derin ağlar yapılandırılmış veri dizilerini (özellikle görüntü) işlemek için kullanılır. Konuşma, ses ve özellikle görüntü verilerinde diğer ağlara göre üstün performans göstermesiyle ayrışır. Görüntü sınıflandırma gibi bilgisayarla görme görevlerinde, giriş görüntülerinden çizgi, daire veya insan yüzü gibi daha karmaşık nesne kalıplarını bulmada çok başarılıdır.

CNN'ler bir dizi halinde üst üste yığılmış birçok konvolüsyon katmanı içerir. Bu sıralı mimari hiyerarşik özellikler öğrenmelerini sağlar. Her katman şekilleri tanıyabilir ve ağ derinleştikçe tanınabilen şekiller daha karmaşık hale gelir. CNN'deki konvolüsyon katmanlarının tasarımı insan görsel korteksinin yapısını yansıtır; görsel korteksimiz de benzer biçimde bir görüntüyü giderek daha karmaşık özellikleri çıkararak katman katman işler.

CNN mimarisi üç temel katmandan oluşur:

#. Konvolüsyon katmanı
#. Havuzlama (Pooling) katmanı
#. Tam bağlantılı (Fully-connected - FC) katman

.. figure:: images/cnn.jpg
      :align: center
      :width: 600 px

      **CNN mimarisine genel bakış.** CNN mimarileri bu yapıyı takip eder ancak her tür için daha fazla katman barındırabilir. Konvolüsyon ve pooling katmanları CNN'lere özgüdür; tam bağlantılı katman, aktivasyon fonksiyonu ve çıktı katmanı ise klasik feed-forward ağlarda da bulunur. Kaynak: [2]

Görüntü verisi ile çalışırken CNN mimarisi RGB ise 3B hacim, gri tonlu ise 1B vektör şeklinde girdi alır. Girdi çeşitli işlemlerden geçirilir ve bir sınıf çıktısı üretilir. İlk katman konvolüsyon katmanıdır; ardından başka konvolüsyon ve pooling katmanları gelebilir. Sonuç çıktısını üreten son katman ise tam bağlantılı katmandır. Her ek konvolüsyon katmanında ağın temsil kapasitesi artar ve görüntünün daha büyük veya daha karmaşık kısımlarını tanıyabilir. İlk katmanlar renk veya kenar gibi basit özellikleri yakalarken, daha derin katmanlar daha büyük ve karmaşık nesneleri ayırt eder. Son katmanlara gelindiğinde nihai FC katmanından önce nesnenin tamamı tanımlanmış olur.


.. rubric:: Model

PyTorch ile örnek bir CNN implementasyonu.

.. literalinclude:: ../code/cnn.py
      :pyobject: CNN

.. rubric:: Eğitim

.. literalinclude:: ../code/cnn.py
      :pyobject: train

.. rubric:: Ek okuma

- `CS231 Konvolüsyonel Ağlar <http://cs231n.github.io/convolutional-networks>`_
- `Deep Learning Kitabı <http://www.deeplearningbook.org/contents/convnets.html>`_


GAN
===
Üretici Çekişmeli Ağ (Generative Adversarial Network - GAN) yeni tensörler (çoğunlukla görüntü, ses vb.) üreten bir ağ türüdür. Mimari içindeki üretici (generator) kısmı ile ayrıştırıcı (discriminator) kısmı sıfır toplamlı (zero-sum) bir oyunda rekabet eder. Üreticinin hedefi, ayrıştırıcının gerçek mi sahte mi olduğunu sınıflandırmaya çalıştığı yeni örnekler üretmektir. İdeal durumda üretici, ayrıştırıcının çıktı için %50 sahte / %50 gerçek olasılığı vermesini sağlayacak kadar ikna edici örnekler üretir.

Şekil [3].

.. image:: images/gan.png
      :align: center

.. rubric:: Model

PyTorch ile örnek bir implementasyon.


.. rubric:: Üretici (Generator)

.. literalinclude:: ../code/gan.py
      :pyobject: Generator
      
.. rubric:: Ayrıştırıcı (Discriminator)

.. literalinclude:: ../code/gan.py
      :pyobject: Discriminator


.. rubric:: Eğitim

.. literalinclude:: ../code/gan.py
      :pyobject: train

.. rubric:: Ek okuma

- `Generative Adversarial Networks <http://guertl.me/post/162759264070/generative-adversarial-networks>`_
- `Deep Learning Kitabı <http://www.deeplearningbook.org/contents/generative_models.html>`_
- `PyTorch DCGAN Örneği <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>`_
- `Orijinal Makale <https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf>`_

MLP
===

Çok Katmanlı Algılayıcı (Multi Layer Perceptron - MLP) yalnızca tam bağlantılı katmanlardan oluşan bir sinir ağıdır. Şekil [5].

.. image:: images/mlp.jpg
      :align: center

.. rubric:: Model

FashionMNIST veri seti üzerinde PyTorch ile bir örnek implementasyon. `Tam Kod <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/mlp.py>`__

1. Ağa giriş 28*28 boyutlu (FashionMNIST'teki 28*28 piksel görüntünün düzleştirilmiş) bir vektördür.
2. 2 tam bağlantılı gizli katman.
3. 10 çıktılı (10 sınıf) çıktı katmanı.

.. literalinclude:: ../code/mlp.py
      :pyobject: MLP

.. rubric:: Eğitim

.. literalinclude:: ../code/mlp.py
      :pyobject: train

.. rubric:: Değerlendirme

.. literalinclude:: ../code/mlp.py
      :pyobject: main


.. rubric:: Ek okuma

YAPILACAK (TODO)


RNN
===

RNN kullanım senaryosu ve temel mimarinin açıklaması.

.. image:: images/rnn.png
      :align: center

.. rubric:: Model

.. literalinclude:: ../code/rnn.py
      :pyobject: RNN

.. rubric:: Eğitim

Bu örnekte girdimiz soyadlarından oluşan bir listedir; her isim değişken uzunlukta tek-sıcak (one-hot) kodlanmış karakter dizisidir. Hedefimiz ise ismin ait olduğu sınıfı (dil) temsil eden indekslerin listesi.

1. Her girdi isim için...
2. Gizli vektörü başlat
3. Karakterler üzerinde döngü kurup sınıfı tahmin et
4. Son karakterin tahminini kayıp fonksiyonuna ver
5. Geri yayılım yap ve ağırlıkları güncelle

.. literalinclude:: ../code/rnn.py
      :pyobject: train

.. rubric:: Ek okuma

- `Jupyter notebook <https://github.com/bfortuner/ml-cheatsheet/blob/master/notebooks/rnn.ipynb>`_
- `Deep Learning Kitabı <http://www.deeplearningbook.org/contents/rnn.html>`_


VAE
===

Autoencoder'lar bir girdi görüntüsünü gizil bir vektöre kodlayıp tekrar çözebilir fakat yeni (görülmemiş) görüntüler üretemez. Varyasyonel Autoencoder'lar (VAE) bu sorunu bir kısıt ekleyerek çözer: Gizil vektör temsili birim Gauss dağılımını modellemelidir. Encoder öğrenilen Gauss'un ortalama ve varyansını döndürür. Yeni bir görüntü üretmek için yeni bir ortalama ve varyans örnekleyip Decoder'a veririz; başka bir deyişle dağılımdan bir "gizil vektör örnekler" ve Decoder'a besleriz. Bu yaklaşım ağın genellemesini iyileştirir ve ezberlemeyi (memorization) önler. Şekil [4].

.. image:: images/vae.png
      :align: center

.. rubric:: Kayıp Fonksiyonu

VAE kaybı yeniden yapılandırma kaybını (ör. Çapraz Entropi, MSE) KL ayrışması (divergence) ile birleştirir.

.. literalinclude:: ../code/vae.py
      :pyobject: vae_loss

.. rubric:: Model

Konvolüsyonel Varyasyonel Autoencoder için PyTorch örnek implementasyonu.

.. literalinclude:: ../code/vae.py
      :pyobject: VAE

.. rubric:: Eğitim

.. literalinclude:: ../code/vae.py
      :pyobject: train

.. rubric:: Ek okuma

- `Orijinal Makale <https://arxiv.org/abs/1312.6114>`_
- `VAE Açıklaması <http://kvfrans.com/variational-autoencoders-explained>`_
- `Deep Learning Kitabı <http://www.deeplearningbook.org/contents/autoencoders.html>`_


.. rubric:: Kaynaklar

.. [1] https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694
.. [2] https://iq.opengenus.org/basics-of-machine-learning-image-classification-techniques/
.. [3] http://guertl.me/post/162759264070/generative-adversarial-networks
.. [4] http://kvfrans.com/variational-autoencoders-explained

.. [#2a] `Applied Deep Learning - Part 3: Autoencoders
<https://towardsdatascience
.com/applied-deep-learning-part-3-autoencoders-1c083af4d798/>`__

.. [#autoenc] `Deep Learning Book - Autoencoders <https://www.deeplearningbook
.org/contents/autoencoders.html/>`__

.. [#tsne] `t-SNE <https://distill.pub/2016/misread-tsne/>`__

.. [#vae] `VAE <https://kvfrans.com/variational-autoencoders-explained/>`__