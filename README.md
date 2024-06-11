# 2mirror_model
Репозиторий содержит программы на языке Python, которые использовались при написании статьи, посвященной математическому моделированию катоптрической системы, состоящей из двух плоских зеркал и видеокамеры. Репозитрий содержит 4 скрипта:


paper_my.py - реализация авторского алгоритма оптимизации оптической системы

paper_nonlinear.py - оптимизация оптической системы с помощью нелинейных методов условной оптимизации из пакета SciPy

paper_linprog.py - модификация авторского алгоритма с решением задачи линейного програмирования с использованием SciPy

paper_3dberry.py - оптимизация одной из существующих стереонасадок с помощью авторской математической модели и возможностей пакета SciPy


Во всех скриптах ряд параметров, которые задают условия задачи:


alpha - половина угла обзора видеокамеры

T_min - минимально допустимое значекние стереобазы

L_1   - максимальная ширина левого зеркала

L_2   - максимальная ширина правого зеркала

u     - габариты видеокамеры


Кроме того, в скриптах paper_my.py и paper_linprog.py можно поменять параметр step, который задает размер шага при изменении параметра b (расстояние от видеокамеры до точки соприкоснования зеркал).

Все скрипты выводят вычисленные углы наклона зеркал в системе координат камеры, значение параметра b и значение стереобазы (base).
