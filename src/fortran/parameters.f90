module parameters
    implicit none
    
    real, parameter :: kappa0 = 1.34e-4     ! diffusion coefficient
    real, parameter :: mu = 2.85e-3         ! mixing coefficient
    real, parameter :: alpha = 3.52         ! attenuation coefficient
    real, parameter :: k_mol = 1.e-7        ! molecular diffusivity
    real, parameter :: T_f = 298.19         ! foundation temperature
    real, parameter :: rho_w = 1027.0       ! water density
    real, parameter :: c_p = 3850.0         ! specific heat capacity of sea water
    
    real, parameter :: z_f = 10.0           ! foundation depth
    real, parameter :: dz0 = 0.1            ! vertical spacing at surface
    real, parameter :: stretch = 1.042155   ! mesh stretch factor
    
    real, parameter :: sigma = 0.8          ! surface suppressivity
    real, parameter :: wind_cutoff = 10.0   ! wind speed cutoff in diffusivity calculation

end module parameters