
!############### MODULE ##############
module krome_ode
contains

  ! *************************************************************
  !  This file has been generated with:
  !  KROME 14.08.dev on 2025-10-08 11:51:50
  !  Changeset cd85309
  !  see http://kromepackage.org
  !
  !  Written and developed by Tommaso Grassi and Stefano Bovino
  !
  !  Contributors:
  !  J.Boulangier, T.Frostholm, D.Galli, F.A.Gianturco, T.Haugboelle,
  !  A.Lupi, J.Prieto, J.Ramsey, D.R.G.Schleicher, D.Seifried, E.Simoncini,
  !  E.Tognelli
  !  KROME is provided "as it is", without any warranty.
  ! *************************************************************

  subroutine fex(neq,tt,nin,dn)
    use krome_commons
    use krome_constants
    use krome_subs
    use krome_cooling
    use krome_heating
    use krome_tabs
    use krome_photo
    use krome_gadiab
    use krome_getphys
    use krome_phfuncs
    use krome_fit
    implicit none
    integer::neq,idust
    real*8::tt,dn(neq),n(neq),k(nrea),krome_gamma
    real*8::gamma,Tgas,vgas,ntot,nH2dust,nd,nin(neq)
    real*8::rr
    integer::i,r1,r2,r3,p1,p2,p3

    n(:) = nin(:)

    nH2dust = 0.d0
    n(idx_CR) = 1.d0
    n(idx_g)  = 1.d0
    n(idx_dummy) = 1.d0

    dn(:) = 0.d0 !initialize differentials
    n(idx_Tgas) = max(n(idx_tgas),2.73d0)
    n(idx_Tgas) = min(n(idx_tgas),1d9)
    Tgas = n(idx_Tgas) !get temperature

    k(:) = coe_tab(n(:)) !compute coefficients

    n(idx_dummy) = 1.d0
    n(idx_g) = 1.d0
    n(idx_CR) = 1.d0
    do i=1,nrea
      r1 = arr_r1(i)
      r2 = arr_r2(i)
      r3 = arr_r3(i)
      p1 = arr_p1(i)
      p2 = arr_p2(i)
      p3 = arr_p3(i)
      rr = k(i)*n(r1)*n(r2)*n(r3)
      dn(r1) = dn(r1) - rr
      dn(r2) = dn(r2) - rr
      dn(r3) = dn(r3) - rr
      dn(p1) = dn(p1) + rr
      dn(p2) = dn(p2) + rr
      dn(p3) = dn(p3) + rr
    end do

    last_coe(:) = k(:)

  end subroutine fex

  !***************************
  subroutine jes(neq, tt, n, j, ian, jan, pdj)
    use krome_commons
    use krome_subs
    use krome_tabs
    use krome_cooling
    use krome_heating
    use krome_constants
    use krome_gadiab
    use krome_getphys
    implicit none
    integer::neq, j, ian, jan, r1, r2, p1, p2, p3, i
    real*8::tt, n(neq), pdj(neq), dr1, dr2, kk,k(nrea),Tgas
    real*8::nn(neq),dn0,dn1,dnn,nH2dust,dn(neq),krome_gamma

    nH2dust = 0.d0
    Tgas = n(idx_Tgas)

    k(:) = last_coe(:) !get rate coefficients

    if(j==1) then
    elseif(j==1) then
      pdj(1) =  &
          -k(1)*n(idx_O)  &
          -k(3)*n(idx_OH)  &
          -k(5)*n(idx_Cj)  &
          -k(8)*n(idx_CO)  &
          -k(9)*n(idx_H2)  &
          +3.d0*k(9)*n(idx_H2)  &
          -9.d0*k(11)*n(idx_H)*n(idx_H)  &
          +3.d0*k(11)*n(idx_H)*n(idx_H)  &
          -4.d0*k(12)*n(idx_H)*n(idx_H2)  &
          -9.d0*k(13)*n(idx_H)*n(idx_H)  &
          +3.d0*k(13)*n(idx_H)*n(idx_H)  &
          -4.d0*k(14)*n(idx_H)*n(idx_H2)
      pdj(2) =  &
          -k(1)*n(idx_O)  &
          +k(3)*n(idx_OH)
      pdj(3) =  &
          +k(1)*n(idx_O)  &
          -k(3)*n(idx_OH)  &
          +k(8)*n(idx_CO)
      pdj(4) =  &
          +k(5)*n(idx_Cj)  &
          +k(8)*n(idx_CO)
      pdj(5) =  &
          -k(8)*n(idx_CO)
      pdj(6) =  &
          +k(3)*n(idx_OH)  &
          -k(9)*n(idx_H2)  &
          +3.d0*k(11)*n(idx_H)*n(idx_H)  &
          -2.d0*k(12)*n(idx_H)*n(idx_H2)  &
          +4.d0*k(12)*n(idx_H)*n(idx_H2)  &
          +3.d0*k(13)*n(idx_H)*n(idx_H)  &
          -2.d0*k(14)*n(idx_H)*n(idx_H2)  &
          +4.d0*k(14)*n(idx_H)*n(idx_H2)
      pdj(7) =  &
          -k(5)*n(idx_Cj)
      pdj(8) =  &
          +k(5)*n(idx_Cj)
    elseif(j==2) then
      pdj(1) =  &
          -k(1)*n(idx_H)  &
          +k(4)*n(idx_H2)
      pdj(2) =  &
          -k(1)*n(idx_H)  &
          -k(2)*n(idx_C)  &
          -k(4)*n(idx_H2)
      pdj(3) =  &
          +k(1)*n(idx_H)  &
          +k(4)*n(idx_H2)
      pdj(4) =  &
          -k(2)*n(idx_C)
      pdj(5) =  &
          +k(2)*n(idx_C)
      pdj(6) =  &
          -k(4)*n(idx_H2)
    elseif(j==3) then
      pdj(1) =  &
          -k(3)*n(idx_H)  &
          +k(7)*n(idx_C)
      pdj(2) =  &
          +k(3)*n(idx_H)
      pdj(3) =  &
          -k(3)*n(idx_H)  &
          -k(7)*n(idx_C)
      pdj(4) =  &
          -k(7)*n(idx_C)
      pdj(5) =  &
          +k(7)*n(idx_C)
      pdj(6) =  &
          +k(3)*n(idx_H)
    elseif(j==4) then
      pdj(1) =  &
          +k(6)*n(idx_Hj)  &
          +k(7)*n(idx_OH)
      pdj(2) =  &
          -k(2)*n(idx_O)
      pdj(3) =  &
          -k(7)*n(idx_OH)
      pdj(4) =  &
          -k(2)*n(idx_O)  &
          -k(6)*n(idx_Hj)  &
          -k(7)*n(idx_OH)
      pdj(5) =  &
          +k(2)*n(idx_O)  &
          +k(7)*n(idx_OH)
      pdj(7) =  &
          +k(6)*n(idx_Hj)
      pdj(8) =  &
          -k(6)*n(idx_Hj)
    elseif(j==5) then
      pdj(1) =  &
          -k(8)*n(idx_H)
      pdj(3) =  &
          +k(8)*n(idx_H)
      pdj(4) =  &
          +k(8)*n(idx_H)
      pdj(5) =  &
          -k(8)*n(idx_H)
    elseif(j==6) then
      pdj(1) =  &
          +k(4)*n(idx_O)  &
          -k(9)*n(idx_H)  &
          +3.d0*k(9)*n(idx_H)  &
          +4.d0*k(10)*n(idx_H2)  &
          -2.d0*k(12)*n(idx_H)*n(idx_H)  &
          -2.d0*k(14)*n(idx_H)*n(idx_H)
      pdj(2) =  &
          -k(4)*n(idx_O)
      pdj(3) =  &
          +k(4)*n(idx_O)
      pdj(6) =  &
          -k(4)*n(idx_O)  &
          -k(9)*n(idx_H)  &
          -4.d0*k(10)*n(idx_H2)  &
          +2.d0*k(10)*n(idx_H2)  &
          -k(12)*n(idx_H)*n(idx_H)  &
          +2.d0*k(12)*n(idx_H)*n(idx_H)  &
          -k(14)*n(idx_H)*n(idx_H)  &
          +2.d0*k(14)*n(idx_H)*n(idx_H)
    elseif(j==7) then
      pdj(1) =  &
          -k(5)*n(idx_H)
      pdj(4) =  &
          +k(5)*n(idx_H)
      pdj(7) =  &
          -k(5)*n(idx_H)
      pdj(8) =  &
          +k(5)*n(idx_H)
    elseif(j==8) then
      pdj(1) =  &
          +k(6)*n(idx_C)
      pdj(4) =  &
          -k(6)*n(idx_C)
      pdj(7) =  &
          +k(6)*n(idx_C)
      pdj(8) =  &
          -k(6)*n(idx_C)
    elseif(j==9) then
    elseif(j==10) then
    elseif(j==11) then

    elseif(j==12) then
    end if

    return
  end subroutine jes

  !*************************
  subroutine jex(neq,t,n,ml,mu,pd,npd)
    use krome_commons
    use krome_tabs
    use krome_cooling
    use krome_heating
    use krome_constants
    use krome_subs
    use krome_gadiab
    implicit none
    real*8::n(neq),pd(neq,neq),t,k(nrea),dn0,dn1,dnn,Tgas
    real*8::krome_gamma,nn(neq),nH2dust
    integer::neq,ml,mu,npd

    Tgas = n(idx_Tgas)
    npd = neq
    k(:) = coe_tab(n(:))
    pd(:,:) = 0d0
    krome_gamma = gamma_index(n(:))

    !d[H_dot]/d[H]
    pd(1,1) =  &
        -k(1)*n(idx_O)  &
        -k(3)*n(idx_OH)  &
        -k(5)*n(idx_Cj)  &
        -k(8)*n(idx_CO)  &
        -k(9)*n(idx_H2)  &
        +3.d0*k(9)*n(idx_H2)  &
        -9.d0*k(11)*n(idx_H)*n(idx_H)  &
        +3.d0*k(11)*n(idx_H)*n(idx_H)  &
        -4.d0*k(12)*n(idx_H)*n(idx_H2)  &
        -9.d0*k(13)*n(idx_H)*n(idx_H)  &
        +3.d0*k(13)*n(idx_H)*n(idx_H)  &
        -4.d0*k(14)*n(idx_H)*n(idx_H2)

    !d[O_dot]/d[H]
    pd(2,1) =  &
        -k(1)*n(idx_O)  &
        +k(3)*n(idx_OH)

    !d[OH_dot]/d[H]
    pd(3,1) =  &
        +k(1)*n(idx_O)  &
        -k(3)*n(idx_OH)  &
        +k(8)*n(idx_CO)

    !d[C_dot]/d[H]
    pd(4,1) =  &
        +k(5)*n(idx_Cj)  &
        +k(8)*n(idx_CO)

    !d[CO_dot]/d[H]
    pd(5,1) =  &
        -k(8)*n(idx_CO)

    !d[H2_dot]/d[H]
    pd(6,1) =  &
        +k(3)*n(idx_OH)  &
        -k(9)*n(idx_H2)  &
        +3.d0*k(11)*n(idx_H)*n(idx_H)  &
        -2.d0*k(12)*n(idx_H)*n(idx_H2)  &
        +4.d0*k(12)*n(idx_H)*n(idx_H2)  &
        +3.d0*k(13)*n(idx_H)*n(idx_H)  &
        -2.d0*k(14)*n(idx_H)*n(idx_H2)  &
        +4.d0*k(14)*n(idx_H)*n(idx_H2)

    !d[C+_dot]/d[H]
    pd(7,1) =  &
        -k(5)*n(idx_Cj)

    !d[H+_dot]/d[H]
    pd(8,1) =  &
        +k(5)*n(idx_Cj)

    !d[H_dot]/d[O]
    pd(1,2) =  &
        -k(1)*n(idx_H)  &
        +k(4)*n(idx_H2)

    !d[O_dot]/d[O]
    pd(2,2) =  &
        -k(1)*n(idx_H)  &
        -k(2)*n(idx_C)  &
        -k(4)*n(idx_H2)

    !d[OH_dot]/d[O]
    pd(3,2) =  &
        +k(1)*n(idx_H)  &
        +k(4)*n(idx_H2)

    !d[C_dot]/d[O]
    pd(4,2) =  &
        -k(2)*n(idx_C)

    !d[CO_dot]/d[O]
    pd(5,2) =  &
        +k(2)*n(idx_C)

    !d[H2_dot]/d[O]
    pd(6,2) =  &
        -k(4)*n(idx_H2)

    !d[H_dot]/d[OH]
    pd(1,3) =  &
        -k(3)*n(idx_H)  &
        +k(7)*n(idx_C)

    !d[O_dot]/d[OH]
    pd(2,3) =  &
        +k(3)*n(idx_H)

    !d[OH_dot]/d[OH]
    pd(3,3) =  &
        -k(3)*n(idx_H)  &
        -k(7)*n(idx_C)

    !d[C_dot]/d[OH]
    pd(4,3) =  &
        -k(7)*n(idx_C)

    !d[CO_dot]/d[OH]
    pd(5,3) =  &
        +k(7)*n(idx_C)

    !d[H2_dot]/d[OH]
    pd(6,3) =  &
        +k(3)*n(idx_H)

    !d[H_dot]/d[C]
    pd(1,4) =  &
        +k(6)*n(idx_Hj)  &
        +k(7)*n(idx_OH)

    !d[O_dot]/d[C]
    pd(2,4) =  &
        -k(2)*n(idx_O)

    !d[OH_dot]/d[C]
    pd(3,4) =  &
        -k(7)*n(idx_OH)

    !d[C_dot]/d[C]
    pd(4,4) =  &
        -k(2)*n(idx_O)  &
        -k(6)*n(idx_Hj)  &
        -k(7)*n(idx_OH)

    !d[CO_dot]/d[C]
    pd(5,4) =  &
        +k(2)*n(idx_O)  &
        +k(7)*n(idx_OH)

    !d[C+_dot]/d[C]
    pd(7,4) =  &
        +k(6)*n(idx_Hj)

    !d[H+_dot]/d[C]
    pd(8,4) =  &
        -k(6)*n(idx_Hj)

    !d[H_dot]/d[CO]
    pd(1,5) =  &
        -k(8)*n(idx_H)

    !d[OH_dot]/d[CO]
    pd(3,5) =  &
        +k(8)*n(idx_H)

    !d[C_dot]/d[CO]
    pd(4,5) =  &
        +k(8)*n(idx_H)

    !d[CO_dot]/d[CO]
    pd(5,5) =  &
        -k(8)*n(idx_H)

    !d[H_dot]/d[H2]
    pd(1,6) =  &
        +k(4)*n(idx_O)  &
        -k(9)*n(idx_H)  &
        +3.d0*k(9)*n(idx_H)  &
        +4.d0*k(10)*n(idx_H2)  &
        -2.d0*k(12)*n(idx_H)*n(idx_H)  &
        -2.d0*k(14)*n(idx_H)*n(idx_H)

    !d[O_dot]/d[H2]
    pd(2,6) =  &
        -k(4)*n(idx_O)

    !d[OH_dot]/d[H2]
    pd(3,6) =  &
        +k(4)*n(idx_O)

    !d[H2_dot]/d[H2]
    pd(6,6) =  &
        -k(4)*n(idx_O)  &
        -k(9)*n(idx_H)  &
        -4.d0*k(10)*n(idx_H2)  &
        +2.d0*k(10)*n(idx_H2)  &
        -k(12)*n(idx_H)*n(idx_H)  &
        +2.d0*k(12)*n(idx_H)*n(idx_H)  &
        -k(14)*n(idx_H)*n(idx_H)  &
        +2.d0*k(14)*n(idx_H)*n(idx_H)

    !d[H_dot]/d[C+]
    pd(1,7) =  &
        -k(5)*n(idx_H)

    !d[C_dot]/d[C+]
    pd(4,7) =  &
        +k(5)*n(idx_H)

    !d[C+_dot]/d[C+]
    pd(7,7) =  &
        -k(5)*n(idx_H)

    !d[H+_dot]/d[C+]
    pd(8,7) =  &
        +k(5)*n(idx_H)

    !d[H_dot]/d[H+]
    pd(1,8) =  &
        +k(6)*n(idx_C)

    !d[C_dot]/d[H+]
    pd(4,8) =  &
        -k(6)*n(idx_C)

    !d[C+_dot]/d[H+]
    pd(7,8) =  &
        +k(6)*n(idx_C)

    !d[H+_dot]/d[H+]
    pd(8,8) =  &
        -k(6)*n(idx_C)

  end subroutine jex

end module krome_ode
