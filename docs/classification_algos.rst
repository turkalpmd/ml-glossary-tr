.. _classification_algos:

=========================
Classification Algorithms (Sınıflandırma Algoritmaları)
=========================

Classification problems (sınıflandırma problemleri), çıktı değişkeni Y'nin belirli kategorilerden (classes) biri olduğu durumları ifade eder: duygu analizi (sentiment analysis) için positive vs negative, görüntü sınıflandırmada (image classification) dog vs cat, tıbbi tanıda (medical diagnosis) disease vs no disease gibi.

Bayesian
=======

Örtüşmeler (overlaps).. (Bu bölüm henüz detaylandırılmamış — katkıya açık.)


Decision Trees (Karar Ağaçları)
==============
.. rubric:: Intuitions (Sezgiler)

Bir decision tree (karar ağacı) veri setini (dataset) ardışık (successive) olarak daha küçük segmentlere böler; hedef değişken (target variable) tek tip olana (pure node) ya da daha fazla bölünemeyene kadar sürer. Greedy (açgözlü) bir algoritmadır; her adımda global optimality (küresel en iyilik) kaygısı olmadan o anda en iyi görünen kararı verir [#mlinaction]_.

Arkasındaki fikir oldukça basittir. Aşağıdaki flowchart (akış diyagramı) decision tree temelli basit bir e‑posta sınıflandırma sistemi gösterir: Adres "myEmployer.com" ise "Email to read when bored" olarak etiketler. Değilse e‑posta "hockey" kelimesini içeriyorsa "Email from friends" olur; aksi halde "Spam: don't read" olarak işaretlenir. Görsel kaynak [#mlinaction]_.

.. image:: images/decision_tree.png
    :align: center
    :scale: 30 %

.. rubric:: Algorithm Explained (Algoritmanın Açıklaması)

Çeşitli decision tree algoritmaları vardır: ID3 (Iterative Dichotomiser 3), C4.5 ve CART (Classification and Regression Trees). Kurulum (construction) adımları benzerdir [#decisiontrees]_:

1. Tüm training instances (eğitim örnekleri) ağacın root'una ata; current node = root.
2. Split criterion (bölme ölçütü) olarak information gain, information gain ratio veya gini coefficient kullanarak split feature ve split value (eşik) seç.
3. Düğümdeki tüm veri örneklerini seçilen özellik ve threshold'a göre partition (böl).
4. Her partition'ı current node'un child node'u olarak işaretle.
5. Her child node için:
    1. Child node “pure” (tek sınıf) ise leaf (yaprak) olarak etiketle ve dur.
    2. Değilse child node'u current node yap ve 2. adıma dön (recursion).

ID3 multiway tree (çok dallı ağaç) oluşturur; her node için target variable (hedef) açısından en yüksek information gain sağlayan categorical feature'ı bulmaya çalışır.

C4.5, ID3'ün halefidir; özelliğin (feature) kategorik olma zorunluluğunu kaldırarak continuous attribute (sürekli özellik) üzerinde dinamik olarak interval'lere bölen bir discrete attribute tanımlar.

CART C4.5'e benzer; farkı binary tree (ikili ağaç) kurması ve regression problem desteği sağlamasıdır [#sklearntree]_.

Ana farklar aşağıdaki tabloda gösterilmiştir:

+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+
|     Dimensions    |         ID3         |                         C4.5                         |                     CART                     |
+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+
|  Split Criterion  |   Information gain  | Information gain ratio (Normalized information gain) | Gini coefficient for classification problems |
+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+
| Types of Features | Categorical feature |           Categorical & numerical features           |       Categorical & numerical features       |
+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+
|  Type of Problem  |    Classification   |                    Classification                    |          Classification & regression         |
+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+
|   Type of Tree    |     Mltiway tree    |                     Mltiway tree                     |                  Binary tree                 |
+-------------------+---------------------+------------------------------------------------------+----------------------------------------------+

.. rubric:: Code Implementation (Kod Uygulaması)

`ID3 <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/decision_tree.py#L87>`__, `C4.5 <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/decision_tree.py#L144>`__ ve `CART <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/decision_tree.py#L165>`__ için object-oriented pattern'lar kullanıldı. Önce üç algoritma için base class (temel sınıf) tanıtılır, ardından CART kodu detaylandırılır.

Önce `TreeNode class <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/decision_tree.py#L7>`__ ve `DecisionTree <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/decision_tree.py#L24>`__ oluşturulur.

.. code-block:: python

    class TreeNode:
        def __init__(self, data_idx, depth, child_lst=[]):
            self.data_idx = data_idx
            self.depth = depth
            self.child = child_lst
            self.label = None
            self.split_col = None
            self.child_cate_order = None

        def set_attribute(self, split_col, child_cate_order=None):
            self.split_col = split_col
            self.child_cate_order = child_cate_order

        def set_label(self, label):
            self.label = label
..

.. code-block:: python

    class DecisionTree()
        def fit(self, X, y):
            """
            X: train data, dimensition [num_sample, num_feature]
            y: label, dimension [num_sample, ]
            """
            self.data = X
            self.labels = y
            num_sample, num_feature = X.shape
            self.feature_num = num_feature
            data_idx = list(range(num_sample))
            # Set the root of the tree
            self.root = TreeNode(data_idx=data_idx, depth=0, child_lst=[])
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                # Check if the terminate criterion has been met
                if node.depth>self.max_depth or len(node.data_idx)==1:
                    # Set the label for the leaf node
                    self.set_label(node)
                else:
                    # Split the node
                    child_nodes = self.split_node(node)
                    if not child_nodes:
                        self.set_label(node)
                    else:
                        queue.extend(child_nodes)
..

CART algoritması binary tree inşa ederken en fazla gain (kazanç) veya en düşük impurity (saf olmama) sağlayacak feature ve threshold'u arar. Split criterion child node'ların impurity kombinasyonudur. Classification için child impurity ölçütü olarak gini coefficient veya information gain; regression için mean-square-error (MSE) veya mean-absolute-error (MAE) kullanılır. Aşağıdaki kod parçası örnektir. Formüller için `scikit-learn documentation <https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation>`__ kısmına bakınız.

.. code-block:: python

    class CART(DecisionTree):

        def get_split_criterion(self, node, child_node_lst):
            total = len(node.data_idx)
            split_criterion = 0
            for child_node in child_node_lst:
                impurity = self.get_impurity(child_node.data_idx)
                split_criterion += len(child_node.data_idx) / float(total) * impurity
            return split_criterion

        def get_impurity(self, data_ids):
            target_y = self.labels[data_ids]
            total = len(target_y)
            if self.tree_type == "regression":
                res = 0
                mean_y = np.mean(target_y)
                for y in target_y:
                    res += (y - mean_y) ** 2 / total
            elif self.tree_type == "classification":
                if self.split_criterion == "gini":
                    res = 1
                    unique_y = np.unique(target_y)
                    for y in unique_y:
                        num = len(np.where(target_y==y)[0])
                        res -= (num/float(total))**2
                elif self.split_criterion == "entropy":
                    unique, count = np.unique(target_y, return_counts=True)
                    res = 0
                    for c in count:
                        p = float(c) / total
                        res -= p * np.log(p)
            return res
..


K-Nearest Neighbor (K-En Yakın Komşu)
==================
.. rubric:: Introduction (Giriş)

K-Nearest Neighbor (KNN) hem classification hem regression için kullanılan supervised learning algoritmasıdır. Prensip: Yeni noktaya (query point) en yakın olan önceden belirlenmiş sayıda (K) training samples bulup bu örneklerden label tahmini yapmak [#sklearnknn]_.

Yeni bir nokta geldiğinde adımlar:

1. Yeni nokta ile tüm training data arasındaki Euclidean distance (Öklid uzaklığı) hesapla
2. En yakın top-K training data seç
3. Regression ise seçilen label'ların ortalamasını al; classification ise en sık (most common / mode) görülen label'ı döndür.

.. rubric:: Code (Kod)

Aşağıda KNN fonksiyonunun Numpy implementasyonu verilmiştir. Ayrıntılar için `code example <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/knn.py>`__.

.. code-block:: python

    def KNN(training_data, target, k, func):
        """
        training_data: all training data point
        target: new point
        k: user-defined constant, number of closest training data
        func: functions used to get the the target label
        """
        # Step one: calculate the Euclidean distance between the new point and all training data
        neighbors= []
        for index, data in enumerate(training_data):
            # distance between the target data and the current example from the data.
            distance = euclidean_distance(data[:-1], target)
            neighbors.append((distance, index))

        # Step two: pick the top-K closest training data
        sorted_neighbors = sorted(neighbors)
        k_nearest = sorted_neighbors[:k]
        k_nearest_labels = [training_data[i][1] for distance, i in k_nearest]

        # Step three: For regression problem, take the average of the labels as the result;
        #             for classification problem, take the most common label of these labels as the result.
        return k_nearest, func(k_nearest_labels)
..


Logistic Regression (Lojistik Regresyon)
===================

Detaylar için bkz. :ref:`logistic regresion <logistic_regression>` (İlgili bölümde açıklamalar ve formüller mevcut.)

Random Forests (Rastgele Ormanlar)
==============

ID3 tabanlı Random Forest Classifier için: `code example <https://github.com/bfortuner/ml-cheatsheet/blob/master/code/random_forest_classifier.py>`__

Boosting
========

Boosting, classification ve regression modellerinin predictive power (tahmin gücü) artırmak için kullanılan güçlü bir yaklaşımdır. Kendi başına bir şey tahmin etmez; zayıf modeller (weak learners) üzerine inşa edilip accuracy (doğruluk) iyileştirmesi yapar. Bu bölümde classification bağlamında açıklanacaktır.

Konuyu anlamak için önce ensembles (topluluk modelleri) ve weighted instances (ağırlıklı örneklerle öğrenme) kavramlarına kısaca değinelim.


.. rubric:: Excurse (Kısa Parantez):
1. **Ensembles (Topluluk Modelleri)**


    Boosting; bagging (ör. Random Forest classifier) ve stacking (bkz. `mlxtend <http://rasbt.github.io/mlxtend/>`__) gibi teknikleri içeren ensemble ailesine dahildir. Fikir "wisdom of the crowd" (kalabalığın bilgeliği) yaklaşımıdır:

    - Tek bir classifier her şeyi bilemez.
    - Birden çok classifier birlikte çok şey bilir.

    Wikipedia buna iyi bir örnektir.

    Önkoşullar:

        - Farklı classifier'lar farklı knowledge (bilgi) taşır.
        - Farklı classifier'lar farklı mistakes (hatalar) yapar.

    İlk koşulu farklı kaynak ve zamanlarda toplanmış farklı datasets kullanarak sağlayabiliriz; pratikte çoğu zaman tek dataset vardır. Bunu cross validation (çapraz doğrulama) ile dolanabiliriz: her fold için bir classifier train edilir.
    İkinci koşul (farklı hata yapma) bu süreçte doğal olarak sağlanmış olur.

    .. figure:: images/grid_search_cross_validation.png
        :align: center
        :width: 400 px

        Ensembles ile cross-validation kullanımı.

    Birden fazla classifier olduğunda sonuçları combine (birleştirme) yöntemi gerekir; farklı ensemble tekniklerinin ortaya çıkma nedeni budur. Farklar; weighted instances kullanıp kullanmama veya sonuçları nasıl birleştirdikleri olabilir. Genel olarak classification için voting, regression için averaging yapılır. Voting / averaging yöntemlerinin weighted gibi varyantları vardır. Bazı yaklaşımlar tüm base-classifier çıktılarını meta classifier için feature olarak kullanıp final prediction üretir (stacking).

2. **Learning with weighted instances (Ağırlıklı Örneklerle Öğrenme)**

    KNN gibi algoritmalar tüm instances'a aynı weight verir (eşit önem). Pratikte örneklerin katkısı farklıdır; örneğin sensör kalitesi değişebilir. Bunu encode etmek için instance ağırlıkları atarız. Yöntem:

    - Classification algoritmasını değiştirmek (maliyetli)
    - Bir instance'ın weight'i n ise onu n kez çoğaltmak (resampling)


Asıl konuya dönersek boosting, classifier'ları (Random Forest'tan farklı olarak paralel değil) ardışık train ederek uygulanır. İlk classifier normal eğitilir. Sonraki classifier'lar önceki modellerin misclassified examples (yanlış sınıfladığı örnekler) üzerine odaklanır. Bunu nasıl sağlarız? Instance ağırlıklarını güncelleyerek. Bir classifier bir örneği yanlış sınıflarsa o örneğin weight'ini artırırız ki sonraki classifier daha fazla dikkat etsin. Doğru örneklerin weight'i genelde değişmez. Boosting bir ensemble tekniğidir ama farklı dataset kullanma kuralını (çeşitlilik için) kısmen bozar; çünkü yanlış sınıflanan örnekleri yeniden kullanmak için tüm veriyi her turda kullanırız. Böylece ilk modelin yanıldığı örnek ikinci veya sonraki modellerde düzeltilir (error reduction iteratif gerçekleşir).


.. figure:: images/boosting_error_iteration.png
    :align: center
    :width: 400 px


    Error decreases with an increasing number of classifiers.

Sıfırdan bir Adaboost implementasyonu (boosting algoritmalarından biri) ve ek açıklamalar: (`python-course.eu <https://python-course.eu/machine-learning/boosting-algorithm-in-python.php/>`__)


Support Vector Machine (SVM)
======================
*Support Vector Machine* (SVM) en popüler supervised learning algoritmalarından biridir; hem classification hem regression için kullanılabilir ancak pratikte çoğunlukla classification içindir. SVM'de her veri örneği *n-dimensional* (n boyutlu) uzayda bir nokta olarak düşünülür; n = feature sayısı, her feature değeri ilgili koordinat değeridir.

Amaç n-boyutlu uzayı farklı classes (sınıflar) olarak ayıran en iyi decision boundary (karar sınırı) yani hyperplane (hiperdüzlem) bulmaktır. En iyi hyperplane, en yakın training point'e (her iki sınıftan) olan mesafeyi (margin) maksimize edendir. Çok sayıda uygun hyperplane olabilir; maksimum margin sağlayanı seçmek genelde daha iyi genelleme verir.

SVM hyperplane'i oluşturmaya katkı veren extreme points (uç noktalar) seçer; bunlara support vectors (destek vektörleri) denir. SVM classifier bu vektörlere dayalı optimal sınırdır.

Aşağıdaki diyagramda mavi ve yeşil iki farklı sınıf gösterilmiştir. *Maximum-margin hyperplane* iki parallel hyperplane (positive / negative hyperplane) arasındaki orta hiperdüzlemdir (kesikli çizgiler). Bu orta düzlem sınıflar arasındaki mesafeyi (margin) maksimize eder.

.. figure:: images/svm.png
      :align: center
      :width: 400 px

      **Support Vector Machine:** Two different categories classified
      using a decision boundary, or hyperplane. Source [#svm]_

SVM iki ana tipte ele alınır:

* **Linear SVM:** Linearly separable (doğrusal ayrılabilir) veri için; tek bir straight line (2D) / hyperplane (yüksek boyut) sınıfları ayırabilir.
* **Non-linear SVM:** Doğrusal olarak ayrılamayan veri için; tek doğru yeterli değildir, kernel trick kullanılır.

dataset has two features, *x1* and *x2*. We want a classifier that can
.. rubric:: Linear SVM

İki feature (*x1*, *x2*) ve iki class (stars, circles) içeren bir dataset düşünelim. Amaç her (*x1*, *x2*) noktasını doğru sınıfa koyan bir classifier. Aşağıdaki şekle bakın.

.. figure:: images/svm_linear.png
      :align: center
      :width: 400 px

      Source [#svm2]_

2 boyutlu uzayda bu iki sınıf bir doğru ile ayrılabilir. Şekilde A, B, C üç olası hyperplane görülüyor. Hangisi optimal? SVM her iki sınıfa ait en yakın noktaları (support vectors) dikkate alır. Support vectors ile hyperplane arasındaki mesafe *margin*'dir. Amaç margin'i maksimize etmek; maksimum margin'li hyperplane = optimal hyperplane. Şekle göre C'nin margin'i A ve B'den büyüktür; dolayısıyla C optimaldir.

.. rubric:: Non-linear SVM

Veri linearly separable değilse tek bir doğru yeterli olmaz. Aşağıdaki şekli inceleyin.

.. figure:: images/svm_nonlinear_1.png
      :align: center
      :width: 300 px

      Source [#svm2]_

the
two features *x* and *y*. For this non-linear data, we will add a third
dimension, *z*. *z* is defined as :math:`z=x^2+y^2`. By adding the third
Circles ile stars ayrımı için ek feature gerekir. Lineer durumda *x*, *y* iki feature yeterliyken; burada üçüncü bir feature *z* ekleyelim: :math:`z = x^2 + y^2`. Böylece veri daha yüksek boyuta (lift) taşınır.

.. figure:: images/svm_nonlinear_2.png
      :align: center
      :width: 300 px

      Source [#svm2]_

dataset into two distinct classes by finding a *linear* hyperplane between
Bu uzayda *z* her zaman pozitiftir (karelerin toplamı). Artık SVM bu yüksek boyutta *linear* bir hyperplane bularak sınıfları ayırabilir.

3 boyutlu uzayda hyperplane bir düzlem gibi görünür. :math:`z=1` dilimine (slice) projeksiyon yaptığımızda 2 boyutlu daire (circle) sınırı elde ederiz.

.. figure:: images/svm_nonlinear_3.png
      :align: center
      :width: 300 px

      Source [#svm2]_

(Yani non-linear veri için yüksek boyutta lineer ayrım; orijinal uzayda :math:`radius=1` çemberi.)

the "kernel trick". The SVM kernel is a function which takes a low
it converts non-linearly separable data to linearly separable data.
Bu hyperplane'i bulmak için *z*'yi elle eklemek zorunda değiliz; SVM "kernel trick" ile düşük boyutlu input'u implicit olarak yüksek boyuta map eder (mapping). Kernel fonksiyonu lineer ayrılamayan veriyi lineer ayrılabilir uzaya dönüştürür.




.. rubric:: References

.. [#sklearnknn] https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
.. [#mlinaction] `Machine Learning in Action by Peter Harrington <https://www.manning.com/books/machine-learning-in-action>`__
.. [#sklearntree] `Scikit-learn Documentations: Tree algorithms: ID3, C4.5, C5.0 and CART <https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart>`__
.. [#sklearnensemble] `Scikit-learn Documentations: Ensemble Method <https://scikit-learn.org/stable/modules/ensemble.html#>`__
.. [#boostingiteration] `Medium-article: what is Gradient Boosting <https://medium.com/analytics-vidhya/what-is-gradient-boosting-how-is-it-different-from-ada-boost-2d5ff5767cb2#>`__
.. [#decisiontrees] `Decision Trees <https://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/>`__
.. [#svm] `Support Vector Machine <https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm>`__
.. [#svm2] `Support Vector Machine <https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/>`__




