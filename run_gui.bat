@echo off
cd /d "%~dp0"
echo =======================================================
echo   OpenArm RH56 Pick-and-Place MuJoCo Simulation Demo
echo =======================================================
echo.
echo Dang khoi chay mo phong trong MuJoCo GUI...
echo.
echo [Camera: bam [ hoac ] de chuyen goc, Esc = Free Camera]
echo   1. Isometric  : Goc nhin 3D toan canh khong gian
echo   2. Front View : Goc nhin chinh dien tu phia truoc
echo   3. Side View  : Goc nhin ngang tu ben man phai
echo   4. Close Grasp: Goc can canh 5 ngon tay kep chat lon sup
echo   5. Overhead   : Goc nhin thang tu tren xuong (2D)
echo   6. Free Camera (Esc): Tu do xoay chuot 360 do
echo.
echo [Dieu khien khi o Free Camera]
echo   - Chuot trai : Xoay goc nhin (Rotate)
echo   - Chuot phai : Phong to / Thu nho (Zoom)
echo   - Chuot giua : Di chuyen (Pan)
echo   - Phim Space : Tam dung / Tiep tuc (Pause/Resume)
echo   - Phim R     : Chay lai episode tu dau
echo.
.\.venv\Scripts\python.exe -u -m simulation.pick_place_demo %*
echo.
echo =======================================================
echo   Mo phong da hoan thanh. Nhan phim bat ky de thoat...
echo =======================================================
pause
