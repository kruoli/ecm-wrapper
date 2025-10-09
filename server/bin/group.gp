FindGroupOrder(p, s, param=0) = {
    A = 0;
    b = 0;

    if (param==0,
            v = Mod(4 * s, p);
            u = Mod(s^2 - 5, p);
            x = u^3;
            A = (3 * u + v) * (v - u)^3 / (4 * x * v) - 2;
            x = x / v^3;
            b = x * (x * (x + A) + 1),
        param==1,
            A = Mod(4 * s^2, p) / 2^64 - 2;
            b = 4 * A + 10,
        param==2,
            E = ellinit([0, Mod(36, p)]);
            [x, y] = ellmul(E, [-3, 3], s);
            x3 = (3 * x + y + 6) / (2 * (y - 3));
            A = -(3 * x3^4 + 6 * x3^2 - 1) / (4 * x3^3);
            b = 1 / (4 * A + 10),
        param==3,
            A = Mod(4 * s, p) / 2^32 - 2;
            b = 4 * A + 10;
    );

    if (param>=0 && param<=3,
            E = ellinit([0, b * A, 0, b^2, 0]);
            ellcard(E),
            0
    )
}
