module krome_ctypes_mod
  use iso_c_binding
  use krome_main
  use krome_user
  implicit none
contains

subroutine krome_init_() bind(C, name="krome_init_")
  call krome_init()
end subroutine krome_init_

subroutine krome_(x, Tgas, dt) bind(C, name="krome_")
  integer, parameter            :: nsp = krome_nmols
  real(c_double), intent(inout) :: x(nsp)
  real(c_double), value         :: Tgas, dt

  call krome(x, Tgas, dt)
end subroutine krome_

subroutine krome_batch_(xall, Tgas, dt, cells) bind(C, name="krome_batch_")
  integer, parameter            :: nsp = krome_nmols
  integer, value                :: cells
  real(c_double), intent(inout) :: xall(nsp,cells)
  real(c_double), intent(in)    :: Tgas(cells)
  real(c_double), value         :: dt
  integer :: i

  do i = i, cells
    call krome(xall(:,i), Tgas(i), dt)
  end do
end subroutine krome_batch_

end module krome_ctypes_mod