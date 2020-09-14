import vtk

import os

os.environ['MESA_GL_VERSION_OVERRIDE'] = "3.2"

# 数据类型vtkPolyData, 生成中心再渲染场景原点的柱体3
cylinder = vtk.vtkCylinderSource()
cylinder.SetHeight(3.0)
cylinder.SetRadius(1.0)
cylinder.SetResolution(10)

# 渲染多边形几何数据，将输入数据转换为几何图元进行渲染辺啊
cylinderMapper = vtk.vtkPolyDataMapper()
cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

cylinderActor = vtk.vtkActor()
cylinderActor.SetMapper(cylinderMapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(cylinderActor)
renderer.SetBackground(0.1, 0.2, 0.4)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(700,700)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)

iren.Initialize()
iren.Start()
