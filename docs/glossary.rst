.. _glossary:

=====================
Glossary (Terimler Sözlüğü)
=====================

Sık kullanılan machine learning (makine öğrenmesi) terimlerinin tanımları. Teknik terimler İngilizce bırakılmış, açıklamalar Türkçe verilmiştir.

.. http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-arbitrary-locations

.. _glossary_accuracy:

Accuracy (Doğruluk)
  Model'in yaptığı doğru prediction (tahmin) yüzdesi. Toplam tahminler içinde doğru olanların oranıdır.

.. _glossary_algorithm:

Algorithm (Algoritma)
  Bir machine learning :ref:`model <glossary_model>` üretmek için kullanılan yöntem / talimatlar dizisi. Örnekler: linear regression, decision trees, support vector machines (SVM), neural networks.

.. _glossary_attribute:

Attribute (Öznitelik)
  Bir observation'ı (gözlem) tanımlayan özellik (ör: color, size, weight). Excel'de sütun başlıklarına karşılık gelir.

.. _glossary_bias_metric:

Bias metric (Bias Metrik)
  Tahminleriniz (predictions) ile gerçek değerler arasındaki ortalama fark. 

  - **Low bias (Düşük bias)** tüm tahminlerin doğru olabileceğini ya da hataların artı/eksi olarak simetrik dağıldığını ve ortalamada küçük olduğunu gösterir.
  - **High bias (Yüksek bias)** (düşük variance ile) model'in underfitting yaptığını ve yanlış architecture kullandığınızı gösterebilir.

.. _glossary_bias_term:

Bias term (Bias Terimi)
  Origin'den geçmeyen pattern'leri temsil etmeyi sağlar. Tüm feature'lar 0 iken output da 0 mı olur sorusuna cevap katar. Genellikle weight'lerle birlikte neuron veya filter'lara eklenir; baz değer ekler.

.. _glossary_categorical_variables:

Categorical Variables (Kategorik Değişkenler)
  Ayrık (discrete) değer kümesine sahip değişkenler. Ordinal (sıra önemli) veya nominal (sıra önemsiz) olabilir.

.. _glossary_classification:

Classification (Sınıflandırma)
  Kategorik (categorical) output tahmini.

  - **Binary classification** iki olası sınıftan birini tahmin eder (ör: e‑posta spam mı değil mi?).
  - **Multi-class classification** birden fazla olası sınıftan birini tahmin eder (ör: resim cat, dog, horse ya da human mı?).

.. _glossary_classification_threshold:

Classification Threshold (Sınıflandırma Eşiği)
  Positive sınıfı kabul edeceğimiz en düşük probability değeri. Örn: diyabet olma olasılığı > %50 ise True döndür, aksi halde False.

.. _glossary_clustering:

Clustering (Kümeleme)
  Unsupervised şekilde verileri benzerliklerine göre gruplara ayırma.

.. _glossary_confusion_matrix:

Confusion Matrix (Karışıklık Matrisi)
  Classification model performance'ını 4 kategoriye ayırarak gösteren tablo.

  - **True Positives (TP)**: Pozitif doğru tahmin.
  - **True Negatives (TN)**: Negatif doğru tahmin.
  - **False Positives (FP)**: Pozitif yanlış tahmin (Type I error).
  - **False Negatives (FN)**: Negatif yanlış tahmin (Type II error).

.. _glossary_continuous_variables:

Continuous Variables (Sürekli Değişkenler)
  Sayısal ölçek üzerinde kesintisiz (range) değer alabilen değişkenler (ör: sales, lifespan).

.. _glossary_convergence:

Convergence (Yakınsama)
  Training sırasında :ref:`loss <glossary_loss>` değerinin iterasyonlar arasında çok az değiştiği durum.

.. _glossary_deduction:

Deduction (Tümdengelim)
  Teoriden başlayıp observation ile test eden üstten aşağı (top‑down) mantık yaklaşımı. Önce hipotez kurulur, sonra verilerle doğrulanır.

.. _glossary_deep_learning:

Deep Learning (Derin Öğrenme)
  Perceptron / Multi Layer Perceptron tabanlı, modern GPU gücü ve büyük veriyle state‑of‑the‑art accuracy elde eden, yapay neural network mimarilerini kapsayan alan. Çok katmanlı yapı sayesinde geleneksel shallow ağlara göre daha yüksek doğruluk ve robust performans sağlar.

.. _glossary_dimension:

Dimension (Boyut)
  ML bağlamında veri setindeki feature sayısını ifade eder. Örn: 28*28*3 resim flatten edildiğinde 3 kanal içerir; house price örneğinde sadece house size varsa 1‑dimensional veri.

.. _glossary_epoch:

Epoch (Epok)
  Algoritmanın tüm dataset'i kaç kez gördüğünü ifade eden tam geçiş (full pass) sayısı.

.. _glossary_extrapolation:

Extrapolation (Ekstrapolasyon)
  Veri seti aralığı dışına tahmin yapma. Training range dışına çıktığınızda model performansı düşebilir.

.. _glossary_false_positive_rate:

False Positive Rate (Yanlış Pozitif Oranı)
  Tanım:

  .. math::

    FPR = 1 - Specificity = \frac{False Positives}{False Positives + True Negatives}

  :ref:`ROC curve <glossary_roc_curve>` üzerindeki x-axis değeridir.

.. _glossary_feature:

Feature (Özellik)
  Bir :ref:`attribute <glossary_attribute>` + değer çiftidir. "Color is blue" bir feature'dır. Excel'de hücre (cell) analojisi yapılabilir.

.. _glossary_feature_selection:

Feature Selection (Özellik Seçimi)
  Model kurulumunda ilgili (relevant) feature'ları seçme süreci (gereksiz/ gürültülü olanları çıkarma).

.. _glossary_feature_vector:

Feature Vector (Özellik Vektörü)
  Çoklu attribute içeren bir observation'ı betimleyen feature dizisi; Excel'de bir satır (row) gibi.

.. _glossary_gradient_accumulation:

Gradient Accumulation (Gradient Biriktirme)
  Büyük batch'i küçük mini-batch'lere bölüp ardışık çalıştırarak efektif büyük batch size simülasyonu yapmak; GPU memory kısıtlarını aşmak için kullanılır.

.. _glossary_hyperparameters:

Hyperparameters (Hiperparametreler)
  Model'in öğrenme sürecini kontrol eden üst düzey ayarlar (ör: learning rate, tree depth, hidden layer sayısı). Training sırasında öğrenilmez, önceden seçilir / ayarlanır.

.. _glossary_induction:

Induction (Tümevarım)
  Observations → teori yönlü bottom‑up mantık yöntemi; tekrar eden X gözlemleri Y sonucunu önerir.

.. _glossary_instance:

Instance (Örnek)
  Dataset içindeki bir data point / row / sample; :ref:`observation <glossary_observation>` ile eş anlamlı.

.. _glossary_label:

Label (Etiket)
  :ref:`supervised learning <glossary_supervised_learning>` bağlamında observation'ın cevap / hedef (target) kısmı. Örn: çiçek sınıflandırmada species label'dır.

.. _glossary_learning_rate:

Learning Rate (Öğrenme Oranı)
  Optimization döngüsündeki weight update adımının büyüklüğü. Yüksek olursa hızlı ilerler ama minimumu overshoot riski artar. Çok düşük olursa kararlı ve hassas ama çok yavaş convergence.

.. _glossary_loss:

Loss (Kayıp)
  Loss = true_value − predicted_value. Düşük olması genelde daha iyi (aşırı overfitting hariç). Accuracy yüzdesel iken loss toplam hata (error) ölçüsüdür; training ve validation set üzerinde ayrı hesaplanır.

.. _glossary_machine_learning:

Machine Learning (Makine Öğrenmesi)
  Mitchell (1997) tanımı: Bir program görev sınıfı T ve performans ölçütü P bağlamında deneyim E ile performansını geliştiriyorsa öğrenmiştir. Basitçe: algoritmalar veriden pattern öğrenip yeni (unseen) veriler için tahmin üretir.

.. _glossary_model:

Model (Model)
  Dataset'i özetleyen weight ve bias gibi parametreleri saklayan yapı; training ile öğrenilir.

.. _glossary_neural_networks:

Neural Networks (Yapay Sinir Ağları)
  Beyin mimarisinden esinlenmiş pattern ve ilişki tanıma amaçlı matematiksel algoritmalar / modeller.

.. _glossary_normalization:

Normalization (Normalizasyon)
  Değerleri belirli bir aralığa / dağılıma dönüştürme; overfitting'i azaltma ve hesaplamayı hızlandırma amacıyla yapılabilir (ör: feature scaling).

.. _glossary_noise:

Noise (Gürültü)
  Veri içindeki pattern'i (örüntü) bulanıklaştıran alakasız veya rastgele bileşenler.

.. _glossary_null_accuracy:

Null Accuracy (Temel Doğruluk)
  Sürekli en sık görülen sınıfı tahmin ederek elde edilen baseline accuracy.

.. _glossary_observation:

Observation (Gözlem)
  Dataset'teki tekil row / sample; :ref:`instance <glossary_instance>` eş anlamlı.

.. _glossary_outlier:

Outlier (Aykırı Değer)
  Veri dağılımından belirgin şekilde sapmış observation.

.. _glossary_overfitting:

Overfitting (Aşırı Öğrenme)
  Model'in training verisine özgü detay ve gürültüyü ezberlemesi; train/validation performansı yüksek, test (genelleme) performansı düşüktür.

.. _glossary_parameters:

Parameters (Parametreler)
  Training sırasında veriden öğrenilen ve optimization ile ayarlanan değerler.

  Örnekler:

  - Neural network weights
  - SVM support vectors
  - Linear / logistic regression coefficients
  

.. _glossary_precision:

Precision (Kesinlik)
  Binary classification'da pozitif tahminlerin ne kadarının doğru olduğunu ölçer.

  .. math::

    P = \frac{True Positives}{True Positives + False Positives}

.. _glossary_recall:

Recall (Duyarlılık / Sensitivity)
  Gerçek pozitiflerin ne kadarını yakaladığımız. Tüm gerçek pozitifler içindeki yakalanan oran.

  .. math::

    R = \frac{True Positives}{True Positives + False Negatives}

.. _glossary_recall_vs_precision:

Recall vs Precision (Recall ve Precision Karşılaştırması)
  Örnek: Beyin MRI'larında tumor (True) var mı yok mu (False) tahmini.

  - **Precision**: Pozitif dediğimiz örneklerin doğruluk yüzdesi. 100 görüntüde sadece 1'ine True der ve o gerçekten True ise precision %100 ama 9 tümörü kaçırdık.
  - **Recall**: Tüm gerçek tümörlerin yüzde kaçını yakaladık. 10 tümörden 1'i yakalandıysa recall %10. Mükemmel recall için 10'unu da bulmalıyız.

.. _glossary_regression:

Regression (Regresyon)
  Sürekli (continuous) output (örn: price, sales) tahmini.

.. _glossary_regularization:

Regularization (Düzenlileştirme)
  Overfitting'e karşı loss function'a complexity penalty (ör: L1, L2) ekleyen teknikler.

.. _glossary_reinforcement_learning:

Reinforcement Learning (Pekiştirmeli Öğrenme)
  Model'in environment ile etkileşip reward sinyalini maksimize etmek için policy öğrenmesi.

.. _glossary_roc_curve:

ROC (Receiver Operating Characteristic) Curve (ROC Eğrisi)
  Farklı :ref:`classification thresholds <glossary_classification_threshold>` için :ref:`true positive rate <glossary_true_positive_rate>` (y‑axis) vs :ref:`false positive rate <glossary_false_positive_rate>` (x‑axis) grafiği. AUC (area under curve) rasgele seçilmiş pozitif ve negatifleri doğru ayırt etme olasılığıdır.

.. _glossary_segmentation:

Segmentation (Segmentasyon)
  Dataset'i benzer örnekleri gruplandıracak şekilde ayrı segmentlere (kümelere) ayırma işlemi.

.. _glossary_specificity:

Specificity (Özgüllük)
  Gerçek negatifleri doğru tahmin etme oranı.

  .. math::

    S = \frac{True Negatives}{True Negatives + False Positives}

.. _glossary_supervised_learning:

Supervised Learning (Denetimli Öğrenme)
  Label'lı veri ile model eğitimi.

.. _glossary_test_set:

Test Set (Test Seti)
  Eğitim sonunda model'in generalization performansını ölçmek için kullanılan ayrı gözlem seti.

.. _glossary_training_set:

Training Set (Eğitim Seti)
  Model parametrelerini öğrenmek için kullanılan gözlem seti.

.. _glossary_transfer_learning:

Transfer Learning (Transfer Öğrenme)
  Bir görev için pre-trained model weight'lerini alıp ikinci bir görevde başlangıç noktası olarak yeniden kullanma yaklaşımı.

.. _glossary_true_positive_rate:

True Positive Rate (TPR)
  :ref:`recall <glossary_recall>` ile aynı.

  .. math::

    TPR = \frac{True Positives}{True Positives + False Negatives}

  :ref:`ROC curve <glossary_roc_curve>` y‑axis değeridir.

.. _glossary_type_1_error:

Type 1 Error (Tip I Hata)
  False Positive. Gerçekte kötü adayın iyi sanılıp işe alınması örneği.

.. _glossary_type_2_error:

Type 2 Error (Tip II Hata)
  False Negative. Gerçekte iyi adayın kaçırılması.

.. _glossary_underfitting:

Underfitting (Yetersiz Öğrenme)
  Model'in verideki önemli pattern çeşitliliğini yakalayamaması; train ve test performansları birlikte düşüktür.

.. _glossary_uat:

Universal Approximation Theorem (Evrensel Yaklaşım Teoremi)
  Yeterli sayıda neuron içeren tek hidden layer'lı bir neural network belirli aralıktaki herhangi bir continuous fonksiyonu yaklaşıklar; farklı aralıklar için yeniden eğitim veya daha fazla neuron gerekebilir.

.. _glossary_unsupervised_learning:

Unsupervised Learning (Denetimsiz Öğrenme)
  Label'sız veri üzerinde pattern / yapı bulma (örn: clustering) eğitimi.

.. _glossary_validation_set:

Validation Set (Doğrulama Seti)
  Training sırasında generalization sinyali sağlayan ayrı gözlem seti; train error düşerken validation error yükselirse overfitting uyarısıdır.

.. _glossary_variance:

Variance (Varyans)
  Aynı observation için model tahminleri arasındaki yayılım.

  - **Low variance**: Tahminler birbirine yakın, iç tutarlılık yüksek.
  - **High variance** (düşük bias ile): Overfitting göstergesi; noise'a aşırı uyum.


.. rubric:: References

.. [1] http://robotics.stanford.edu/~ronnyk/glossary.html
.. [2] https://developers.google.com/machine-learning/glossary
