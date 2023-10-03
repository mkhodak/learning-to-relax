function k = sor(A, b, x, omega, tol)

D = diag(diag(A));
M = D/omega + tril(A,-1);

k = 0;
r = b - A*x;
norm0 = norm(r);
while norm(r) / norm0 >= tol
  k = k + 1;
  x = x + M\r;
  r = b - A*x;
  if (norm(r)/norm0 < tol)
    return
  end
end
