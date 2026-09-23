# WC-LCRL-STC — спецификация (как получена, нормативный текст)

WC-LCRL-STC заменяет онлайн-аутентификацию всех постоянных весов в RP-STC на
предварительно построенный coefficient-RS enrollment и связывает его с
фактически использованными во время inference проекциями через поздний
случайный coefficient bridge.

Одновременно протокол сохраняет fresh Ligero для активаций, маршрутов,
нелинейностей и fixed-point witness, но устраняет повторные semantic forward
passes посредством недоверенного message cache.

## 0.1. Одноразовый enrollment модели

Все постоянные линейные карты группируются по ширине выхода:
`n ∈ {128, 5120, 8192, 16384, 202048}`.

Внутри каждой группы входные координаты всех матриц конкатенируются и
разбиваются на блоки длины `B = 15360`.

Для каждого блока `a` и выходной координаты `j` строится полином

    F_{n,a,j}(X) = Σ_{b<B} W_{n,a,b,j} X^b + Σ_{h<λ} z_{n,a,j,h} X^{B+h}

где `λ = 1024`, `K_w = B + λ = 16384`. Первые `B` коэффициентов содержат
exact decoded weights, остальные `λ` — независимые случайные маски.

Каждый полином вычисляется на RS-домене размера `N_w = 32768`, codewords
сохраняются column-major и связываются Merkle root.

Enrollment manifest фиксирует: digest исходного GGUF; точный алгоритм
декодирования каждого qtype; shapes и tensor order; transpose/permutation
flags; tied views; block offsets; zero-filled tails; RMSNorm gains.

Enrollment обязан доказывать или детерминированно проверять, что low
coefficients действительно совпадают с canonical decoded model.

## 0.2. R1: inference, fresh commitments и cache

Prover выполняет один exact semantic forward pass. Во время прохода он:

1. вычисляет реальные layer outputs;
2. фиксирует активации, маршруты, nonlinear и rescale witnesses в fresh
   Ligero commitments;
3. сохраняет canonical fresh message rows и pad metadata в недоверенный
   message cache;
4. включает model/enrollment identifier в statement;
5. фиксирует все semantic outputs, которые позднее будут проверяться.

После завершения R1 все outputs всех claims одной output width неизменяемы.
Только теперь Fiat–Shamir генерирует по одному общему challenge
`ρ^(n) ∈ F^n` для каждой output width `n`. Общий `ρ^(n)` позволяет
упаковать разные матрицы одной ширины в общие coefficient blocks; это
безопасно только потому, что все соответствующие outputs зафиксированы до
его генерации.

## 0.3. R2: реальные проекции и projected masks

Для каждой fixed-weight map вычисляется semantic projection
`P_trace = W ρ^(n)`. Это должна быть именно та переменная `P_trace`/`y`,
которую используют дальнейшие terminal constraints; отдельная несвязанная
копия запрещена.

Для каждого enrollment block вычисляются projected mask coefficients:
`π_{n,a,h} = Σ_j z_{n,a,j,h} ρ^(n)_j`.

R2 commitment фиксирует: `P_trace`; projected masks `π`; routed
auxiliaries; остальные phase-2 witness values. R2 должен получать исходные
semantic messages из cache и не выполнять повторный полный forward pass.

## 0.4. Поздний coefficient bridge

Только после фиксации R2 Fiat–Shamir генерирует: независимый `α_{n,a}` для
каждого packed block; `q_w = 40` различных RS-точек `η_1..η_40`; поздние
routed challenges `σ, λ`.

Coefficient vector каждого блока:

    U_{n,a,k} = P_trace[n,a,k]   (0 ≤ k < B)
              = π_{n,a,k−B}      (B ≤ k < K_w)

Для gain blocks low coefficients берутся из тех же semantic gain
variables, которые используют RMSNorm claims.

Агрегат: `c_k = Σ_{n,a} α_{n,a} U_{n,a,k} + Σ_{a∈G} α_a U^g_{a,k}`,
`0 ≤ k < K_w`; значения `v_ℓ = Σ_{k<K_w} c_k η_ℓ^k`, `1 ≤ ℓ ≤ 40`.

`c` и `v_ℓ` фиксируются в R3/transcript до последующих qLin challenges.
Fresh qLin доказывает оба соотношения. Verifier открывает enrollment
codewords в тех же точках `η_ℓ` и проверяет

    v_ℓ = Σ_n Σ_a α_{n,a} Σ_j ρ^(n)_j F_{n,a,j}(η_ℓ) + Σ_{a∈G} α_a F^g_a(η_ℓ).

Если semantic projection ≠ Wρ, после фиксации R2 возникает ненулевой
полином степени < K_w; он исчезает при агрегировании α с вероятностью
≤ 1/p, либо обнуляется во всех 40 точках с вероятностью

    H_40 = C(K_w−1, 40) / C(N_w, 40) = C(16383,40)/C(32768,40)
         ≈ 8.859073161·10^−13.

## 0.5. Проверка routed experts

Сохраняются отношения `P = Wρ`, `Q = MP`, `H = X⊙Q`, `Σ_k H_{t,k} = Y_t ρ`.
Первое заменяется coefficient bridge; остальные — fresh constraints и
selective R3 (f_y, f_u, f_p = f_u·f_y, Σ f_p = Σ λ_t Q σ_k). Маршрут
приватен: все 128 expert enrollments открываются по одинаковому шаблону.

## 0.6. Fresh Ligero без semantic replay

После R3 — существующие fresh проверки (qIRS, qLin, p_0, LogUp,
range/rescale, route/permutation), но witness берётся из message cache.
Ровно два прохода по fresh cache: (1) fused qIRS/qLin/p0; (2) opening
re-encode для 54 fresh columns. Merkle levels/paths из R1/R2/R3
сохраняются; opening stage не хеширует все fresh codewords заново.

## 0.7. Что даёт ускорение

Структурно удаляются: persistent-weight rows из online qIRS; persistent
LF1B rows из online qLin; полное RS-перекодирование весов перед openings;
повторные semantic forward passes; полный R3 semantic replay. Остаются:
один post-R1 проход для Wρ; enrollment masks; coefficient bridge; чтение
40 column-major enrollment columns; два прохода по cached fresh messages.

## 0.8. Инвариант корректности

Для каждой fixed-weight операции существует непрерывная цепочка

    canonical GGUF → enrolled F → Wρ → P_trace → terminal semantic
    constraints → committed output.

Компилятор обязан завершаться ошибкой, если хотя бы одно persistent поле,
gain, embedding, router, expert matrix или LM-head weight не входит ровно
в одну такую цепочку.

## 0.9. Поправки 2026-09-23 (обязательны для верификатора)

1. **Открытие в точке `η_ℓ` привязано к её индексу.** Путь Merkle колонки
   проверяется по индексу `η_ℓ`: порядок хеширования на каждом уровне
   выводится из индекса, длина пути равна глубине дерева над `N_w` листьями,
   индекс лежит в `[0, N_w)`. Валидный путь другой колонки — REJECT. Иначе
   `F(−X)` проходит под root от `F(X)`: на запрос `i` отвечает колонка
   `i + N_w/2`, и каждое уравнение моста выполняется.

2. **Доверенный якорь — идентичность enrollment, а не его root.** Root
   фиксирует колонки, но не их прочтение: `B` и `λ` приходят из
   доказательства, и сдвиг границы между весами и масками при том же `K_w`
   сохраняет root. Внешняя политика задаёт

       ID = blake3(домен v1, правило укладки, root, manifest digest,
                   B, λ, N_w, [(n, сегменты строк по claims в порядке
                   claims, всего строк, блоков, строк нулевого заполнения)]
                   по возрастанию n)

   (домен и правило укладки — с префиксом длины u64 LE; root и manifest
   digest — ровно по 32 байта, без префикса; все целые, включая число групп
   и число сегментов перед каждым списком, — u64 LE). Верификатор пересчитывает
   `ID` из root, manifest и геометрии доказательства и из раскладки,
   выведенной из собственного набора claims, и сравнивает с политикой.
   Кодирование одинаково в Python (`wc_bridge.enrollment_identity`) и Rust
   (`wc_enrollment_identity`); общие тестовые векторы закрепляют его. Порог
   `q_w ≥ ⌈0.416·t⌉` и якорь плотных весов (root блока W) остаются
   независимыми.
