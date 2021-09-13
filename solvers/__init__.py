from .SRSolver import SRSolver
from .SRSolver_cate import SRSolver_cate


def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_cate(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver_cate(opt)
    else:
        raise NotImplementedError

    return solver