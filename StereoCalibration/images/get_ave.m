function z = get_ave(a, b, Depth)
    temp0 = Depth(a(1) : b(1), a(2) : b(2));
    [x, y] = size(temp0);
    z = sum(temp0, 'all') / x / y;
end