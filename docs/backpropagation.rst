.. _backpropagation:

===============
Backpropagation
===============

.. contents:: :local:

Backpropagation (geri yayılım) sürecinin hedefleri basittir: ağdaki her bir weight (ağırlık) toplam error'a (hata, cost) ne kadar katkıda bulunuyorsa ona orantılı biçimde ayarlansın. Her bir weight'in katkısını (error gradient) iteratif olarak azaltabilirsek sonuçta iyi tahminler (predictions) üreten bir weight setine ulaşırız.


Chain Rule (Zincir Kuralı) Hatırlatma
=====================================

Forward propagation (ileri yayılım) iç içe (nested) denklemler dizisi gibi görülebilir. Feed forward işlemini böyle düşünürsek backpropagation yalnızca :ref:`chain_rule` uygulayarak cost'un (kayıp) herhangi bir ara değişkene göre :ref:`derivative` (türevini) bulmaktır. Aşağıdaki forward propagation fonksiyonu verilsin:

.. math::

  f(x) = A(B(C(x)))

A, B ve C farklı katmanlardaki activation function'lardır. Chain rule (zincir kuralı) ile :math:`f(x)` fonksiyonunun :math:`x`'e göre türevi kolayca hesaplanır:

.. math::

  f'(x) = f'(A) \cdot A'(B) \cdot B'(C) \cdot C'(x)

B'ye göre türev nasıl? :math:`B(C(x))` ifadesini sabitmiş gibi düşünebilir, yerini sembolik bir B ile doldurup normal şekilde B'ye göre türevi alabilirsiniz.

.. math::

  f'(B) = f'(A) \cdot A'(B)

Bu basit teknik herhangi bir fonksiyon içindeki tüm ara değişkenlere uygulanabilir ve her değişkenin toplam output (çıktı) üzerindeki etkisini (duyarlılığını) hassas biçimde belirlememizi sağlar.



Chain Rule Uygulaması
=====================

Şimdi chain rule kullanarak ağdaki herhangi bir weight'e göre cost'un türevini (gradient) hesaplayalım. Chain rule bize her weight'in toplam error'a katkısını ve error'u azaltmak için weight'in hangi yönde güncelleneceğini söyler. Aşağıda bir tahmin (prediction) yapmak ve total error (cost) hesaplamak için gereken denklemler yer alıyor:

.. image:: images/backprop_ff_equations.png
    :align: center

Tek nöronlu bir ağ için total cost şu şekilde yazılabilir:

.. math::

  Cost = C(R(Z(X W)))

Chain rule ile Cost'un W weight'ine göre türevini kolayca buluruz.

.. math::

  C'(W) &= C'(R) \cdot R'(Z) \cdot Z'(W) \\
        &= (\hat{y} -y) \cdot R'(Z) \cdot X

Artık herhangi bir weight'e göre cost türevini hesaplayan bir formülümüz olduğuna göre yukarıdaki basit (toy) yapay sinir ağı örneğine dönelim.

.. image:: images/simple_nn_diagram_zo_zh_defined.png
    :align: center

Cost'un :math:`W_o`'ya göre türevi nedir?

.. math::

  C'(W_O) &= C'(\hat{y}) \cdot \hat{y}'(Z_O) \cdot Z_O'(W_O) \\
          &= (\hat{y} - y) \cdot R'(Z_O) \cdot H

Peki :math:`W_h` için? Bunun için fonksiyonda geriye giderek chain rule'u yinelemeli (recursive) uygularız ve Wh terimini içeren kısma ulaşırız.

.. math::

  C'(W_h) &= C'(\hat{y}) \cdot O'(Z_o) \cdot Z_o'(H) \cdot H'(Z_h) \cdot Z_h'(W_h) \\
          &= (\hat{y} - y) \cdot R'(Z_o) \cdot W_o \cdot R'(Z_h) \cdot X

Bir de eğlence için: Ağımız 10 hidden layer (gizli katman) içerse ilk weight :math:`w_1` için cost türevi nasıl görünür?

.. math::

  C'(w_1) = \frac{dC}{d\hat{y}} \cdot \frac{d\hat{y}}{dZ_{11}} \cdot \frac{dZ_{11}}{dH_{10}} \cdot \\ \frac{dH_{10}}{dZ_{10}} \cdot \frac{dZ_{10}}{dH_9} \cdot \frac{dH_9}{dZ_9} \cdot \frac{dZ_9}{dH_8} \cdot \frac{dH_8}{dZ_8} \cdot \frac{dZ_8}{dH_7} \cdot \frac{dH_7}{dZ_7} \cdot \\ \frac{dZ_7}{dH_6} \cdot \frac{dH_6}{dZ_6} \cdot \frac{dZ_6}{dH_5} \cdot \frac{dH_5}{dZ_5} \cdot \frac{dZ_5}{dH_4} \cdot \frac{dH_4}{dZ_4} \cdot \frac{dZ_4}{dH_3} \cdot \\ \frac{dH_3}{dZ_3} \cdot \frac{dZ_3}{dH_2} \cdot \frac{dH_2}{dZ_2} \cdot \frac{dZ_2}{dH_1} \cdot \frac{dH_1}{dZ_1} \cdot \frac{dZ_1}{dW_1}

Deseni (pattern) görüyor musunuz? Ağ derinleştikçe cost türevlerini hesaplamak için gereken işlem sayısı artar. Ayrıca tekrar eden (redundant) hesaplamalar var: Her katmanın cost türevi kendisinden önce hesaplanmış terimlerin sonuna iki yeni terim ekliyor. Bu tekrarları önleyip yaptığımız işi saklayabileceğimiz (cache) bir yol olsa?



Memoization ile Hesabı Saklamak
================================

Memoization bilgisayar bilimlerinde "aynı şeyi tekrar tekrar hesaplama" prensibidir. Önceden hesaplanan sonuçları saklayarak (cache) aynı fonksiyonu yeniden hesaplamaktan kaçınırız. Backpropagation bir tür recursive (özyinelemeli) süreç olduğundan bunu hızlandırmak için idealdir. Aşağıdaki türev denklemlerindeki pattern'e bakın.


.. image:: images/memoization.png
    :align: center

Bu katmanların her biri aynı türevleri yeniden hesaplıyor! Her weight için uzun türev ifadeleri yazmak yerine memoization ile ağ boyunca error geri yayılırken ara sonuçları saklayabiliriz. Bunun için backpropagation için gerekli tüm hesapları kapsayan 3 denklem (aşağıda) tanımlarız. Matematik aynıdır; fakat bu denklemler hangi hesapları yaptığımızı izlemek için pratik bir kısaltma (shorthand) sağlar.

.. image:: images/backprop_3_equations.png
    :align: center

Önce output layer error (çıktı katmanı hatası) hesaplanır ve bir önceki hidden layer'a (gizli katmana) aktarılır. Sonra bu gizli katmanın error değeri hesaplanır ve ondan önceki gizli katmana geri gönderilir (propagate backward). Böylece ağın başına kadar devam eder. Geri giderken her katmanda 3. formülü uygulayarak cost'un o katmanın weight'lerine göre türevini buluruz. Bu türev ağırlıkları toplam cost'u azaltacak yönde nasıl güncelleyeceğimizi söyler.

.. note::

  *Layer error* terimi cost'un bir katmanın *input*'una göre türevi anlamına gelir. Şu soruyu yanıtlar: O katmanın girdisi (activation öncesi girdi) değiştiğinde cost çıktısı nasıl değişir?

.. rubric:: Output Layer Error (Çıktı Katmanı Hatası)

Output layer error hesaplamak için cost'un output layer input'u :math:`Z_o`'ya göre türevini bulmalıyız. Bu bize son katmanın weight'lerinin toplam error'u nasıl etkilediğini gösterir. Türev şöyledir:

.. math::

  C'(Z_o) = (\hat{y} - y) \cdot R'(Z_o)

Gösterimi sadeleştirmek için makine öğrenmesi uygulayıcıları genelde :math:`(\hat{y}-y) * R'(Z_o)` ifadesini :math:`E_o` ile değiştirir. Böylece output layer error formülü:

.. math::

  E_o = (\hat{y} - y) \cdot R'(Z_o)

.. rubric:: Hidden Layer Error (Gizli Katman Hatası)

Hidden layer error için cost'un gizli katman input'u :math:`Z_h`'ya göre türevi gerekir.

.. math::

  C'(Z_h) = (\hat{y} - y) \cdot R'(Z_o) \cdot W_o \cdot R'(Z_h)

Tekrarı önlemek için yukarıdaki :math:`E_o` terimini yerine koyarak sadeleştirilmiş hidden layer error denklemini elde ederiz:

.. math::

  E_h = E_o \cdot W_o \cdot R'(Z_h)

Bu formül backpropagation'ın özüdür. Mevcut katmanın error'unu hesaplayıp ağırlıklandırılmış hata değerini bir önceki katmana geri geçiririz; ilk hidden layer'a ulaşana kadar sürer. Bu sırada her weight için cost türevini kullanarak weight güncelleriz.

.. rubric:: Cost'un Herhangi Bir Weight'e Göre Türevi

Output layer weight'i :math:`W_o` için cost türevi formülüne dönelim.

.. math::

  C'(W_O) = (\hat{y} - y) \cdot R'(Z_O) \cdot H

İlk kısmı output layer error :math:`E_o` ile değiştirebileceğimizi biliyoruz. Buradaki H gizli katman activation'ını temsil eder.

.. math::

  C'(W_o) = E_o \cdot H

Ağımızdaki herhangi bir weight'e göre cost türevini bulmak için ilgili katmanın error'unu kendi input'u (bir önceki katman output'u / activation'ı) ile çarpmamız yeterlidir.

.. math::

  C'(w) = CurrentLayerError \cdot CurrentLayerInput

.. note::

  *Input* burada weighted input (Z) değil, bir önceki katmandan gelen activation anlamındadır.

.. rubric:: Özet

Backpropagation'ın temelini oluşturan 3 nihai denklem aşağıdadır.

.. image:: images/backprop_final_3_deriv_equations.png
    :align: center

Yukarıdaki basit sinir ağı örneği üzerinde süreç görselleştirilmiştir.

.. image:: images/backprop_visually.png
    :align: center

Kod Örneği (Code Example)
=========================

.. literalinclude:: ../code/nn_simple.py
    :language: python
    :lines: 17-41



.. rubric:: Kaynaklar

.. [1] Example
