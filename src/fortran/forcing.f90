module forcing
    use grid, only: nx, ny, nz
    implicit none
    
    ! Surface forcing
    real :: flux_shortwave(nx, ny)      ! downward shortwave radiative flux
    real :: flux_longwave(nx, ny)       ! longwave radiative flux
    real :: flux_latent(nx, ny)         ! latent heat flux
    real :: flux_sensible(nx, ny)       ! sensible heat flux
    real :: wind_u(nx, ny)              ! zonal wind speed
    real :: wind_v(nx, ny)              ! meridional wind speed

    ! Initial ocean temperature field
    real :: T_init(nx,ny,nz)

end module forcing