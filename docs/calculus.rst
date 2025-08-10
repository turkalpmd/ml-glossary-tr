.. _calculus:

========
Calculus
========

.. contents:: :local:


.. _introduction:

Introduction (Giriş)
====================

Fonksiyonların zaman (ya da giriş) değişimine göre nasıl değiştiğini anlamak için temel düzeyde Calculus (türev - derivative) bilmek ve belirli bir aralık boyunca birikmiş toplam miktarı hesaplamak için integral (integral) kavramını bilmek gerekir. Calculus dili, fonksiyonların özelliklerini (properties) kesin biçimde ifade etmenizi ve davranışlarını daha iyi anlamanızı sağlar.

Klasik bir Calculus dersi genellikle elde uzun ve yorucu hesaplamalar yapmayı içerir; ancak bilgisayar gücü sayesinde süreç daha eğlenceli olabilir. Bu bölümde makine öğrenmesi (machine learning) kavramlarını anlamak için bilmeniz gereken temel Calculus fikirleri özetlenmiştir.


.. _derivative:

Derivatives (Türevler)
======================

Bir derivative (türev) iki şekilde tanımlanabilir:

  #. Anlık değişim oranı (instantaneous rate of change – Physics/fiziksel yorum)
  #. Bir doğrunun belirli bir noktadaki eğimi (slope at a point – Geometric/geometrik yorum)

Her ikisi aynı prensibi temsil eder; burada geometrik tanımı üzerinden ilerlemek daha sezgiseldir.


Geometric Definition (Geometrik Tanım)
--------------------------------------

Geometride slope (eğim) bir doğrunun ne kadar dik olduğunu gösterir. Şu soruya cevap verir: :math:`x` belirli bir miktar değiştiğinde :math:`y` ya da :math:`f(x)` ne kadar değişir?

.. image:: images/slope_formula.png
    :align: center

Bu tanım ile iki nokta arasındaki eğim kolayca hesaplanır. Peki iki nokta arasındaki eğim yerine, doğrunun tek bir noktasındaki eğim nedir dersek? Bu durumda doğrudan bir "rise-over-run" (yükselme/boyuna oran) ifadesi yok. İşte derivative bu soruyu yanıtlamamıza yardım eder.

Bir derivative, bir doğru üzerindeki tek bir noktada *instantaneous rate of change* (anlık değişim oranı) yani slope'u hesaplayabileceğimiz bir ifade verir. Türevi bulduktan sonra, doğrunun diğer tüm noktalarındaki eğimi de hesaplayabilirsiniz.


Taking the Derivative (Türevi Alma)
-----------------------------------

Aşağıdaki grafikte :math:`f(x) = x^2 + 3` olsun.

.. image:: images/calculus_slope_intro.png
    :align: center

 (1,4) ile (3,12) noktaları arasındaki slope (eğim) şöyle olur:

.. math::

  slope = \frac{y2-y1}{x2-x1} = \frac{12-4}{3-1} = 4

Peki (1,4) noktasındaki eğimi nasıl buluruz? Bu noktanın tam yerel eğimini (değişim oranını) nasıl ortaya çıkarırız?

Bir yöntem en yakın iki noktayı bulup onların eğimlerinin ortalamasını almaktır. Ancak Calculus daha basit ve kesin bir yol sunar: derivative hesaplamak. Bu, iki nokta seçmemize benzer; fakat bu kez :math:`x`'e sonsuz küçük (infinitesimal) bir uzaklıkta hayali bir nokta alır ve aradaki slope'u hesaplarız.

Bu sayede derivative şu soruyu cevaplar: :math:`x` değerini inanılmaz derecede küçük bir miktar artırırsak :math:`f(x)` nasıl değişir? Yani derivative, birbirine sonsuz küçük uzaklıkta olan iki nokta arasındaki slope'u *tahmin* etmemize yardım eder (yeterince küçük ama hesap yapabilecek kadar büyük bir fark düşünün).

Matematiksel olarak bu sonsuz küçük artışı bir limit ile ifade ederiz. Limit, giriş (input) belirli bir değere yaklaşırken fonksiyon çıktısının yaklaştığı değerdir. Burada hedef değer, slope'unu istediğimiz özel noktadır.


Step-by-step (Adım Adım)
------------------------

Bir derivative hesaplamak normal slope hesaplamaya benzer; fark, noktamız ile ona sonsuz küçük uzaklıktaki başka nokta arasındaki slope'u almamızdır. Bu infinitesimal mesafeyi :math:`h` ile gösteririz. Adımlar:

1. Given the function:

.. math::

  f(x) = x^2

2. :math:`x`'i çok küçük bir :math:`h` kadar artır (:math:`h = Δx`)

.. math::

  f(x + h) = (x + h)^2

3. Slope formülünü uygula

.. math::

  \frac{f(x + h) - f(x)}{h}

4. Denklemi sadeleştir

.. math::

  \frac{x^2 + 2xh + h^2 - x^2}{h} \\

  \frac{2xh+h^2}{h} = 2x+h

5. :math:`h`'yi 0'a götür (limit, :math:`h -> 0`)

.. math::

  {2x + 0} = {2x}

Bu ne anlama gelir? :math:`f(x) = x^2` fonksiyonu için herhangi bir noktadaki slope'un :math:`2x` olduğunu gösterir. Genel formül:

.. math::

  \lim_{h\to0}\frac{f(x+h) - f(x)}{h}


.. rubric:: Code


Herhangi bir :math:`f(x)` fonksiyonunun derivative'ını sayısal olarak hesaplayan bir kod yazalım. :math:`f(x)=x^2` için test edip gerçek türev :math:`2x`'e yakın değer üretmesini bekliyoruz.

::

  def get_derivative(func, x):
      """Compute the derivative of `func` at the location `x`."""
      h = 0.0001                          # step size
      return (func(x+h) - func(x)) / h    # rise-over-run

  def f(x): return x**2                   # some test function f(x)=x^2
  x = 3                                   # the location of interest
  computed = get_derivative(f, x)
  actual = 2*x

  computed, actual   # = 6.0001, 6        # pretty close if you ask me...


Genel olarak tam (closed-form) `derivative formulas <https://www.teachoo.com/9722/1227/Differentiation-Formulas/category/Finding-derivative-of-a-function-by-chain-rule/>`_ kullanmak tercih edilir; ancak küçük bir adım :math:`h` için rise-over-run hesaplayarak sayısal (numerical) türev alabileceğinizi unutmayın.


Machine Learning Use Cases (Kullanım Alanları)
----------------------------------------------

Machine learning optimizasyon (optimization) problemlerinde derivative kullanır. *Gradient descent* gibi algoritmalar bir objective (hedef) fonksiyonu (ör. accuracy ya da error) minimize/maksimize etmek için weight'lerin artırılıp azaltılacağını derivative (gradient) bilgisiyle belirler. Türev ayrıca eğrinin tangent line (teğet) yaklaşımıyla doğrusal (linear) olarak yaklaştırılmasını sağlar; doğrusal fonksiyonların slope'u sabittir. Sabit slope ile hedef değere (class label) yaklaşmak için hangi yönde (aşağı / yukarı) ilerleyeceğimize karar verebiliriz.



.. _chain_rule:

Chain Rule (Zincir Kuralı)
==========================

Chain rule, composite functions (bileşik fonksiyonlar) için türev hesaplama formülüdür. Composite function: başka fonksiyon(lar) içinde tanımlı fonksiyon.

How It Works (Nasıl Çalışır)
----------------------------

Bir composite function :math:`f(x) = A(B(x))` verildiğinde, :math:`f(x)`'in türevi :math:`A`'nın :math:`B(x)`'e göre türevi ile :math:`B`'nin :math:`x`'e göre türevinin çarpımıdır.

.. math::

  \mbox{composite function derivative} = \mbox{outer function derivative} * \mbox{inner function derivative}

Örneğin aşağıdaki gibi bir composite function :math:`f(x)` verilsin:

.. math::

  f(x) = h(g(x))

Chain rule'a göre :math:`f(x)`'in türevi:

.. math::

  \frac{df}{dx} = \frac{dh}{dg} \cdot \frac{dg}{dx}


Step-by-step (Adım Adım)
------------------------

Varsayalım :math:`f(x)` iki fonksiyondan oluşuyor: :math:`h(x) = x^3` ve :math:`g(x) = x^2`. O halde:

.. math::

  \begin{align}
  f(x) &= h(g(x)) \\
       &= (x^2)^3 \\
  \end{align}

:math:`f(x)`'in türevi:

.. math::

  \begin{align}
  \frac{df}{dx} &=  \frac{dh}{dg} \frac{dg}{dx} \\
                &=  \frac{dh}{d(x^2)} \frac{dg}{dx}
  \end{align}


.. rubric:: Steps (Adımlar)

1. İç fonksiyonun (inner) türevini hesapla :math:`g(x) = x^2`

.. math::

  \frac{dg}{dx} = 2x

2. Dış fonksiyonun (outer) türevini hesapla :math:`h(x)=x^3`; iç fonksiyonu :math:`x^2` bir placeholder :math:`b` ile göster.

.. math::

  \frac{dh}{db} = 3b^2

3. Placeholder :math:`b` yerine inner function :math:`g(x)` koy.

.. math::
  \begin{gathered}
  3(x^2)^2 \\
  3x^4
  \end{gathered}

4. İki türevin çarpımını al

.. math::

  3x^4 \cdot 2x = 6x^5


Multiple Functions (Birden Fazla İç Fonksiyon)
----------------------------------------------

Yukarıdaki örnekte tek iç fonksiyonlu bir composite function varsaydık. Chain rule daha derin iç içe fonksiyonlar için de uygulanabilir:

.. math::

  f(x) = A(B(C(x)))

Chain rule'a göre türev:

.. math::

  \frac{df}{dx} = \frac{dA}{dB} \frac{dB}{dC} \frac{dC}{dx}

Bu türev :math:`f'` gösterimiyle de yazılabilir:

.. math::

  f' = A'(B(C(x)) \cdot B'(C(x)) \cdot C'(x)


.. rubric:: Steps (Adımlar)


:math:`f(x) = A(B(C(x)))` fonksiyonu için şunları varsayalım:

.. math::

  \begin{align}
  A(x) & = sin(x) \\
  B(x) & = x^2 \\
  C(x) & = 4x
  \end{align}

Bu fonksiyonların türevleri:

.. math::

  \begin{align}
  A'(x) &= cos(x) \\
  B'(x) &= 2x \\
  C'(x) &= 4
  \end{align}

:math:`f(x)`'in türevini şu formülle hesaplarız:

.. math::

  f'(x) = A'( (4x)^2) \cdot B'(4x) \cdot C'(x)

Türevleri yerine koyup ifadeyi sadeleştiririz:

.. math::

  \begin{align}
  f'(x) &= cos((4x)^2) \cdot 2(4x) \cdot 4 \\
        &= cos(16x^2) \cdot 8x \cdot 4 \\
        &= cos(16x^2)32x
  \end{align}





.. _gradient:

Gradients (Gradyanlar)
======================

Bir gradient (gradyan) çok değişkenli (multivariable) fonksiyonların partial derivative (kısmi türev) değerlerini tutan bir vektördür. Birden fazla bağımsız değişkeni olan fonksiyonlarda belirli bir noktadaki slope'u hesaplamaya yardım eder. Daha karmaşık bu slope'u hesaplamak için her değişkeni tek tek izole eder, diğerlerini sabit tutarken fonksiyonun türevini alırız. Her iterasyonda bir partial derivative elde eder ve gradient vektörüne yerleştiririz.


Partial Derivatives (Kısmi Türevler)
------------------------------------

İki veya daha fazla değişkenli fonksiyonlarda partial derivative, bir değişkenin diğerleri sabitken aldığı türevdir. :math:`x`'i değiştirip diğerlerini sabit tutarsak :math:`f(x,z)` nasıl değişir? Bu bir partial derivative'dır. Sonra :math:`z` için aynı işlemi yaparız. Kısmi türevleri gradient vektöründe saklarız; bu vektör çok değişkenli fonksiyonun toplam türevi hakkında bilgi verir.


Step-by-step (Adım Adım)
------------------------

Çok değişkenli (multivariable) bir fonksiyonun gradient'ini hesaplama adımları:

1. Given a multivariable function

.. math::

  f(x,z) = 2z^3x^2

2. Calculate the derivative with respect to :math:`x`

.. math::

  \frac{df}{dx}(x,z)

3. :math:`2z^3` ifadesini sabit (constant) :math:`b` ile değiştir

.. math::

  f(x,z) = bx^2

4. :math:`b` sabitken türevini al

.. math::

  \begin{align}
  \frac{df}{dx} & = \lim_{h\to0}\frac{f(x+h) - f(x)}{h} \\
                & = \lim_{h\to0}\frac{b(x+h)^2 - b(x^2)}{h} \\
                & = \lim_{h\to0}\frac{b((x+h)(x+h)) - bx^2}{h} \\
                & = \lim_{h\to0}\frac{b((x^2 + xh + hx + h^2)) - bx^2}{h} \\
                & = \lim_{h\to0}\frac{bx^2 + 2bxh + bh^2 - bx^2}{h} \\
                & = \lim_{h\to0}\frac{2bxh + bh^2}{h} \\
                & = \lim_{h\to0} 2bx + bh \\
  \end{align}

 :math:`h —> 0` iken...

  2bx + 0

5. :math:`x`'e göre türev için :math:`2z^3` ifadesini geri koy

.. math::

  \begin{align}
  \frac{df}{dx}(x,z) &= 2(2z^3)x \\
                     &= 4z^3x
  \end{align}

6. :math:`z`'ye göre türevi hesaplamak için benzer adımları uygula

.. math::

  \frac{df}{dz}(x,z) = 6x^2z^2

7. Kısmi türevleri gradient vektöründe sakla

.. math::

   \nabla f(x,z)=\begin{bmatrix}
       \frac{df}{dx} \\
       \frac{df}{dz} \\
      \end{bmatrix}
   =\begin{bmatrix}
       4z^3x \\
       6x^2z^2 \\
      \end{bmatrix}


Directional Derivatives (Yönlü Türevler)
----------------------------------------

Diğer önemli kavram directional derivative'dır. Çok değişkenli fonksiyonlarda partial derivative hesaplarken her bağımsız değişkende sonsuz küçük artışların etkisini inceleriz. Her bir değişkeni artırmak fonksiyon çıktısını slope yönünde değiştirir.

Peki yön değiştirmek istersek? 3B arazide kuzeye doğru ilerlediğimizi ve gradient'in de bulunduğumuz noktada kuzeyi gösterdiğini varsayalım. Ama güneybatı (southwest) yönüne gitmek istersek o yöndeki eğimi (steepness) nasıl buluruz? Directional derivative, gradient'in gösterdiğinden farklı bir yönde hareket edersek slope'u hesaplamamıza yardım eder.


.. rubric:: Math (Matematik)

Directional derivative, :math:`f` fonksiyonunun gradient'i ile yönü temsil eden bir unit vector (birim vektör) :math:`\vec{v}`'nin dot product'ının [11]_ alınmasıyla hesaplanır. Unit vector her eksende ne oranda ilerlemek istediğimizi gösterir. Çıktı, girdi :math:`\vec{v}` yönünde küçükçe itilirse :math:`f`'in ne kadar değişeceğini gösteren bir scalar (skaler) değerdir.

Elinizde :math:`f(x,y,z)` fonksiyonu olsun ve aşağıdaki vektör boyunca directional derivative hesaplamak isteyin [2]_: 

.. math::

 \vec{v}=\begin{bmatrix}
   2 \\
   3 \\
   -1  \\
  \end{bmatrix}


Yukarıda anlatıldığı gibi gradient ile yön vektörünün dot product'ını alırız:

.. math::

   \begin{bmatrix}
     \frac{df}{dx} \\
     \frac{df}{dy} \\
     \frac{df}{dz} \\
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
       2 \\
       3 \\
       -1  \\
    \end{bmatrix}


Dot product'ı şu şekilde yeniden yazabiliriz:

.. math::

  \nabla_\vec{v} f = 2 \frac{df}{dx} + 3 \frac{df}{dy} - 1 \frac{df}{dz}

Bu anlamlıdır; çünkü :math:`\vec{v}` yönündeki küçük bir itme x yönünde iki, y yönünde üç ve z yönünde −1 (geri) küçük itmenin bileşimidir.


Useful Properties (Faydalı Özellikler)
--------------------------------------

Derin öğrenmede (deep learning) gradient ile ilgili özellikle yararlı iki özellik:

  #. Gradient her zaman fonksiyonun en hızlı artış (greatest increase) yönünü gösterir (`açıklama <https://betterexplained.com/articles/understanding-pythagorean-distance-and-the-gradient>`_)
  #. Yerel maksimum (local maximum) veya yerel minimum (local minimum) noktalarında gradient sıfırdır.






.. _integrals:

Integrals (İntegraller)
=======================

Bir :math:`f(x)` fonksiyonunun integral'i, grafiğinin altında kalan alanın hesaplanmasıdır. :math:`x=a` ile :math:`x=b` arasında kalan alan şöyle gösterilir:

.. math::

   A(a,b) = \int_a^b f(x) \: dx.

.. image:: images/integral_definition.png
   :align: center

:math:`A(a,b)` alanı üstte :math:`f(x)` fonksiyonu, altta :math:`x`-ekseni ve yanlarda :math:`x=a`, :math:`x=b` dikey çizgileriyle sınırlıdır. :math:`x=a`, :math:`x=b` noktaları limits of integration (integrasyon sınırları) olarak adlandırılır. :math:`\int` sembolü Latince "summa" (toplam) kelimesinden gelir. İntegral, bu iki sınır arasındaki :math:`f(x)` değerlerinin "toplamı"dır (sürekli toplam).

*Integral function* :math:`F(c)`, üst limit (upper limit) :math:`c`'ye göre alanı veren fonksiyondur:

.. math::

  F(c) \equiv \int_0^c \! f(x)\:dx\,.

Bu formülde iki değişken ve bir sabit vardır. Girdi değişkeni :math:`c` üst limittir. *Integration variable* :math:`x`, :math:`0`'dan :math:`c`'ye tarama (sweep) yapar. :math:`0` alt limittir (lower limit). Başlangıç noktasının 0 seçilmesi keyfidir (arbitrary).

Integral function :math:`F(c)`, :math:`f(x)` grafiği altındaki alan hakkında önceden hesaplanmış (precomputed) bilgiyi içerir. Derivative function :math:`f'(x)`, tüm :math:`x` değerleri için grafiğin slope (eğim) bilgisini verirken, :math:`F(c)` tüm olası integrasyon limitleri için grafiğin altındaki alan bilgisini verir.

:math:`x=a` ile :math:`x=b` arasındaki alan integral fonksiyonundaki *değişim* olarak hesaplanır:

.. math::

   A(a,b) = \int_a^b \! f(x)\:dx
   	=  F(b)-F(a).

.. image:: images/integral_as_change_in_antriderivative.png
   :align: center



Computing Integrals (İntegralleri Yaklaşık Hesaplama)
----------------------------------------------------

:math:`x=a` ile :math:`x=b` arasında :math:`f(x)` altındaki toplam alanı, bölgeyi genişliği :math:`h` olan dikey şeritlere ayırıp her dikdörtgenin alanını toplayarak yaklaşıklarız. Aşağıdaki şekilde :math:`f(x)=x^2` için :math:`x=1` ile :math:`x=3` arasındaki alanın, :math:`h=0.5` genişlikli 4 dikdörtgen ile nasıl yaklaşıklandığı gösterilir.

.. image:: images/integral_as_rectangular_strips.png
   :align: center

Genellikle approximation (yaklaşım) daha doğru olsun diye :math:`h` küçük seçilir. Aşağıda integrasyon yapan örnek kod var.

::

  def get_integral(func, a, b):
      """Compute the area under `func` between x=a and x=b."""
      h = 0.0001               # width of small rectangle
      x = a                    # start at x=a
      total = 0
      while x <= b:            # continue until x=b
          total += h*func(x)   # area of rect is base*height
          x += h
      return total

  def f(x): return x**2                    # some test function f(x)=x^2
  computed = get_integral(f, 1, 3)
  def actualF(x): return 1.0/3.0*x**3   
  actual = actualF(3) - actualF(1)
  computed, actual    # = 8.6662, 8.6666   # pretty close if you ask me...


Integral function'ları türev formülleri (derivative rules) ve biraz tersine mühendislik (reverse engineering) ile bulabilirsiniz. :math:`f(x)` için bir integral function bulmak demek :math:`F'(x)=f(x)` olacak :math:`F(x)` fonksiyonunu bulmak demektir. :math:`f(x)` verilip integral function :math:`F(x)` sorulduğunu varsayalım:

.. math::

   F(x) = \int \! f(x)\: dx.

Bu, türevi :math:`f(x)` olan bir :math:`F(x)` bulma problemidir:

.. math::

  F'(x) = f(x).


Örneğin :math:`\int \!x^2\:dx` belirsiz integrali bulunmak istensin. Bu şu :math:`F(x)` fonksiyonunu aramakla eşdeğerdir:

.. math::

  F'(x) = x^2.

Türev kurallarını hatırlayarak :math:`F(x)`'in :math:`x^3` terimi içermesi gerektiğini tahmin ederiz; kubik terimin türevi ikinci dereceden terim verir. O halde aranan fonksiyon :math:`F(x)=c x^3` biçimindedir. :math:`c` sabitini şu denklemi sağlayacak şekilde seç:

.. math::

  F'(x) = 3cx^2 = x^2.

:math:`3c=1` çözülürse :math:`c=\frac{1}{3}` bulunur ve integral function:

.. math::

  F(x) = \int x^2 \:dx = \frac{1}{3}x^3 + C.

Doğrulama: :math:`\frac{d}{dx}\left[\frac{1}{3}x^3 + C\right] = x^2`.

Ek olarak çeşitli integral doğrulamaları için şu `formulas <https://www.teachoo.com/5643/728/Integration-Formulas---Trig--Definite-Integrals-Properties-and-more/category/Miscellaneous/>`_ kaynağına bakabilirsiniz.


Applications of Integration (İntegrasyon Uygulamaları)
-----------------------------------------------------

İntegral hesapları çok sayıda bilim alanında kullanılır. Burada probability (olasılık) ile ilgili birkaç örnek verelim.


Computing Probabilities (Olasılık Hesaplama)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sürekli (continuous) bir random variable :math:`X`, probability density function (olabilirlik yoğunluk fonksiyonu) :math:`p(x)` ile tanımlanır. :math:`p(x)` eğrisi altındaki toplam alan 1 olan pozitif bir fonksiyondur:

.. math::

	  p(x) \geq 0, \forall x 
    \qquad
	   \textrm{and}
	   \qquad
	   \int_{-\infty}^\infty p(x)\; dx = 1.

:math:`X`'in :math:`a` ile :math:`b` arasında değer alma olasılığı şu integral ile verilir:

.. math::

	 \textrm{Pr}(a \leq X \leq b)
   =
	 \int_a^b p(x)\; dx.

Böylece integrasyon kavramı sürekli random variable'lar içeren probability theory için merkezîdir.

Ayrıca integrali random variable'ın bazı karakteristik özelliklerini hesaplamak için kullanırız. *Expected value (beklenen değer / ortalama)* ve *variance (varyans)* :math:`X` için davranışına dair temel iki özelliktir.


Expected Value (Beklenen Değer / Mean)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bir random variable :math:`X`'in *expected value*'ı şu formülle hesaplanır:

.. math::

  \mu
	% \equiv \mathbb{E}_X[X]
	= \int_{-\infty}^\infty x\, p(x).

Expected value, random variable :math:`X`'in ortalama olarak hangi değeri vereceğini gösterir; aynı zamanda *average* veya *mean* olarak da adlandırılır.



Variance (Varyans)
~~~~~~~~~~~~~~~~~~

Random variable :math:`X` için *variance* şu şekilde tanımlanır:

.. math::

   \sigma^2
	 % \equiv  \mathbb{E}_X\!\big[(X-\mu)^2\big] 
	 = \int_{-\infty}^\infty (x-\mu)^2 \, p(x).

Varyans formülü :math:`X`'in expected value :math:`\mu`'dan karesel uzaklığının beklentisini hesaplar. :math:`\sigma^2` ya da :math:`var(X)` değerleri :math:`X` sonuçlarının ne kadar yaygın (spread) ya da kümelenmiş (clustered) olduğunu gösterir. Küçük varyans sonuçların :math:`\mu` etrafında sıkı kümelendiğini, büyük varyans genişçe yayıldığını ifade eder. Varyansın kareköküne *standard deviation (standart sapma)* denir ve :math:`\sigma` ile gösterilir.

Expected value :math:`\mu` ve variance :math:`\sigma^2` probability ve statistics alanlarında herhangi bir random variable'ı karakterize etmemizi sağladığı için merkezî kavramlardır. Expected value *central tendency* (merkezi eğilim), variance ise *dispersion* (dağılım) ölçüsüdür. Fiziğe aşina olanlar expected value'u dağılımın *centre of mass*'ı, variance'ı ise *moment of inertia* (atalet momenti) gibi düşünebilir.




.. rubric:: References

.. [1] https://en.wikipedia.org/wiki/Derivative
.. [2] https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/directional-derivative-introduction
.. [3] https://en.wikipedia.org/wiki/Partial_derivative
.. [4] https://en.wikipedia.org/wiki/Gradient
.. [5] https://betterexplained.com/articles/vector-calculus-understanding-the-gradient
.. [6] https://www.mathsisfun.com/calculus/derivatives-introduction.html
.. [7] http://tutorial.math.lamar.edu/Classes/CalcI/DefnOfDerivative.aspx
.. [8] https://www.khanacademy.org/math/calculus-home/taking-derivatives-calc/chain-rule-calc/v/chain-rule-introduction
.. [9] http://tutorial.math.lamar.edu/Classes/CalcI/ChainRule.aspx
.. [10] https://youtu.be/pHMzNW8Agq4?t=1m5s
.. [11] https://en.wikipedia.org/wiki/Dot_product
