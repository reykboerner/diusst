module grid
    use parameters, only: mu, z_f, stretch, dz0
    implicit none

    integer i, j, k     ! i = zonal index, j = meridional index, k = vertical index

    ! horizontal domain
    integer, parameter :: nx = 10, ny = 10
    real :: x(nx), y(ny)

    ! vertical domain
    integer, parameter :: nz = 42
    real :: z(nz)           ! vertical coordinate
    real :: dz(nz-2)        ! vertical grid spacing

    real :: dv1(nz)         ! first spatial derivative on stretched vertical grid
    real :: dv2(nz)         ! second spatial derivative on stretched vertical grid
    real :: relax_rate(nz)  ! rate of temperature relaxation

    contains

    subroutine init_vertical_mesh
        z(1) = dz0          ! Atmospheric dummy point
        z(2) = 0.0          ! Sea surface
        
        do k = 3, nz
            z(k) = -dz0*(1 - stretch**(k-1))/(1 - stretch)
        end do

        dz = z(2:nz-1) - z(1:nz-2)

        ! stretched mesh derivatives
        do k=1,nz
            dv1(k) = 1/(log(stretch)*(dz0/(1 - stretch) + z(k)))
            dv2(k) = -1/(log(stretch)*(dz0/(1 - stretch) + z(k))**2)
        end do
        
        ! mixing term prefactor   
        do k=2,nz-1
            relax_rate(k) = mu/abs(z(k) - z_f)*(z(k+1) - z(k))
        end do
    end subroutine init_vertical_mesh

end module grid