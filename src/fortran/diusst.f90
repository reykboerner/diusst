module diusst

    ! ---------------------------------------------
    ! DiuSST, a conceptual diurnal warm layer model
    ! 
    ! Info: github.com/reykboerner/diusst
    ! Author: Reyk Boerner
    ! Date: 17 June 2025
    !
    ! Equation numbers refer to Boerner et al. 2025
    ! https://doi.org/10.5194/gmd-18-1333-2025
    ! ---------------------------------------------    

    use grid
    use forcing

    implicit none

    real :: T(nx, ny, nz)           ! sea temperature field
    real :: SST(nx, ny)             ! sea surface temperature field
    
    contains

    subroutine step
        ! This subroutine steps DiuSST forward in time (following Eq. A1).
        
        use parameters

        real :: dt, dt_atmo         ! DiuSST time step, atmospheric forcing time step
        integer :: n, niter         ! index and number of time iterations
        
        real :: cfl_array(nz)       ! CFL numbers
        real :: wind2               ! squared wind speed
    
        real :: kappa(nz)           ! diffusivity profile
        real :: dkappadz(nz)        ! spatial derivative of diffusivity profile
        real :: radflux(nz)         ! radiative flux profile
        real :: T_old(nz)           ! column temperature buffer
        
        ! set radiation flux value of dummy point
        radflux(1) = 0.0

        ! iterate over horizontal grid
        do j = 1, ny
            do i = 1, nx

                ! get squared wind speed
                wind2 = min(wind_cutoff**2, wind_u(i,j)**2 + wind_v(i,j)**2)

                ! determine DiuSST time step based on CFL condition
                ! and number of DiuSST iterations within forcing time step
                do k = 2, nz-1
                    cfl_array(k) = 2*(k_mol + kappa0*wind2*( &
                        1 + sigma*(abs(z(k)/z_f)-1)))/dz(k)**2
                end do
                dt = min(dt_atmo, 0.95/maxval(cfl_array))
                niter = ceiling(dt_atmo/dt)
                dt = dt_atmo/niter
                
                ! compute diffusivity profile and its derivative
                do k = 2, nz-1
                    kappa(k) = k_mol + kappa0*wind2*(1 + sigma*(abs(z(k)/z_f) - 1))
                    dkappadz(k) = - kappa0*wind2*sigma/z_f
                end do

                ! compute radiation flux profile
                do k = 2, nz
                    radflux(k) = flux_shortwave(i,j)*exp(alpha*z(k))
                end do
                radflux(2) = radflux(2) &
                    + flux_longwave(i,j) + flux_sensible(i,j) + flux_latent(i,j)

                ! iteratine in time
                T_old = T_init(i,j,:)
                T(i,j,:) = T_old
                do n = 1, niter
                    do k = 2, nz-1
                        T(i,j,k) = T_old(k) + dt*( &
                            kappa(k)*((T_old(k+1) - 2*T_old(k) + T_old(k-1))*dv1(k)**2 &
                            + (T_old(k+1) - T_old(k-1))/2*dv2(k)) &
                            + dkappadz(k)*((T_old(k-1) - T_old(k+1))/2*dv1(k)) & 
                            + relax_rate(k)*(T_old(k)-T_f) &
                            + ((radflux(k+1) - radflux(k)) * dv1(k))/(rho_w*c_p) &
                        )
                    end do
                    T_old = T(i,j,k)
                end do

                ! update atmospheric dummy point
                T(i,j,1) = T(i,j,2)

            end do
        end do

        ! update SST
        SST = T(i,j,2)

    end subroutine step

end module diusst