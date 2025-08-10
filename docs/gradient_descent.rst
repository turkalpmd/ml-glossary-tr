.. _gradient_descent:

============================
Gradient Descent (Gradyan İnişi)
============================

Gradient Descent bir optimization algorithm olup bir fonksiyonu minimize etmek için her iterasyonda negatif gradient (en dik iniş) yönünde küçük adımlar atar. Machine learning bağlamında model :ref:`parameters <glossary_parameters>` (ör: :doc:`linear_regression` katsayıları, neural network :ref:`weights <nn_weights>`) değerlerini güncellemek için kullanılır.


Introduction (Giriş)
============

Alttaki 3‑boyutlu grafiği bir cost function yüzeyi olarak düşünün. Amacımız sağ üstteki dağ (yüksek cost) noktasından sol alttaki koyu mavi deniz (düşük cost) bölgesine inmektir. Oklar herhangi bir noktadan en hızlı azalış yönünü (negative gradient, steepest descent) gösterir; yani cost'u en hızlı düşüren yön. `Source <http://www.adalta.it/Pages/-GoldenSoftware-Surfer-010.asp>`_

.. image:: images/gradient_descent.png
    :align: center

Dağın tepesinden başlar, negatif gradient yönünde ilk adımı atarız. Sonra yeni noktadaki gradient'i tekrar hesaplayıp yine ters yönde (negatif gradient) ilerleriz. Bu iterative süreç alt seviyeye (global minimum mümkünse) ya da daha fazla inemediğimiz bir local minimum noktasına kadar sürer. `image source <https://youtu.be/5u0jaA3qAGk>`_.

.. image:: images/gradient_descent_demystified.png
    :align: center

Learning Rate (Öğrenme Oranı)
=============

Bu adımların büyüklüğüne *learning rate* denir. Yüksek learning rate her adımda daha fazla mesafe kat ettirir ama minimumu overshoot etme (aşma) riskini artırır. Çok düşük learning rate ise daha hassas (precise) fakat yavaştır; gradient sık hesaplandığı için yön daha güvenilir olur ancak convergence süresi uzar.


Cost Function (Maliyet Fonksiyonu)
=============

:ref:`cost_function` modelimizin belirli parametreler altında hangi doğrulukla tahmin yaptığını ölçer. Bu fonksiyonun kendine ait bir yüzeyi (eğrisi) ve her noktada gradient'i vardır; eğimin yönü parametreleri hangi yönde güncellememiz gerektiğini söyler.


Step-by-step (Adım Adım)
============

Şimdi bu cost function için gradient descent çalıştıralım. Kontrol ettiğimiz iki parametre: :math:`m` (weight) ve :math:`b` (bias). Her birinin prediction üzerindeki etkisi olduğundan partial derivative kullanırız; cost function'ın :math:`m` ve :math:`b`'ye göre türevlerini hesaplayıp gradient vektöründe saklarız.

.. rubric:: Math (Matematik)

Cost function:

.. math::

  f(m,b) =  \frac{1}{N} \sum_{i=1}^{N} (y_i - (mx_i + b))^2

Gradient hesaplanışı:

.. math::

  f'(m,b) =
     \begin{bmatrix}
       \frac{df}{dm}\\
       \frac{df}{db}\\
      \end{bmatrix}
  =
     \begin{bmatrix}
       \frac{1}{N} \sum -2x_i(y_i - (mx_i + b)) \\
       \frac{1}{N} \sum -2(y_i - (mx_i + b)) \\
      \end{bmatrix}

Gradient'i bulmak için veri noktaları üzerinden yeni :math:`m` ve :math:`b` ile geçip partial derivative toplamlarını hesaplarız. Elde edilen gradient mevcut parametre konumunda cost surface eğimini ve hangi yöne ilerlememiz gerektiğini gösterir; adım büyüklüğü learning rate ile ölçeklenir.


.. rubric:: Code (Kod)

::

  def update_weights(m, b, X, Y, learning_rate):
      m_deriv = 0
      b_deriv = 0
      N = len(X)
      for i in range(N):
          # Calculate partial derivatives
          # -2x(y - (mx + b))
          m_deriv += -2*X[i] * (Y[i] - (m*X[i] + b))

          # -2(y - (mx + b))
          b_deriv += -2*(Y[i] - (m*X[i] + b))

      # We subtract because the derivatives point in direction of steepest ascent
      m -= (m_deriv / float(N)) * learning_rate
      b -= (b_deriv / float(N)) * learning_rate

      return m, b


.. rubric:: References (Kaynaklar)

.. [1] http://ruder.io/optimizing-gradient-descent
