function [q] = comp_flux_gen(flux,res,u,Grid,BC)
% author: Marc Hesse
% date: 22 Feb 2019
% Description:
% Computes the fuxes on the interior from flux(u) and reconstructs the
% fluxes on the boundary faces from the residuals in the adjacent boundary
% cells, res(u). 

% Input:
% flux = anonymous function computing the flux (correct in the interior)
% res = anonymous function computing the residual 
% u = vector of 'flux potential' (head, temperature,electric field,...)
% Grid = structure containing pertinent information about the grid
% BC = structure containing pertinent information about BC's
% 
% Output:
% q = correct flux everywhere
%
% Example call:


%% Compute interior fluxes
q = flux(u);


%% Compute boundary fluxes
% 1) Identify the faces and cells on the boundary
dof_cell = [BC.dof_dir;BC.dof_neu];
dof_face = [BC.dof_f_dir;BC.dof_f_neu];
% 2) Determine sign of flux: Convention is that flux is positive in
%    coordinate direction. So the boundary flux, qb is not equal to q*n,
%    were n is the outward normal! 
sign = ismember(dof_face,Grid.dof_f_xmin) - ismember(dof_face,Grid.dof_f_xmax);

% 3) Compute residuals and convert them to bnd fluxes
q(dof_face) = res(u,dof_cell).*Grid.V(dof_cell)./Grid.A(dof_face);


