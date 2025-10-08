! Thin wrapper to expose clean C-style functions for Python ctypes
subroutine krome_init_c() bind(C, name="krome_init_c")
  use krome_main
  use krome_user
  implicit none
  call krome_init()
end subroutine krome_init_c

subroutine krome_c(x, Tgas, dt) bind(C, name="krome_c")
  use krome_main
  implicit none
  real*8, intent(inout) :: x(:)
  real*8, intent(in)    :: Tgas, dt
  call krome(x, Tgas, dt)
end subroutine krome_c

