#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#define CGAL_SEGMENTATION_BENCH_GRAPHCUT
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/copyleft/cgal/mesh_to_polyhedron.h>
#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
#include <CGAL/internal/Surface_mesh_segmentation/Alpha_expansion_graph_cut.h>
#include <igl/matrix_to_list.h>
#include <igl/per_face_normals.h>
#include <igl/PI.h>
#include <algorithm>


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
bool sdf_segmentation(const Eigen::MatrixXd&V, const Eigen::MatrixXi &F, Eigen::MatrixXd& C)
{
  Eigen::MatrixXi BI;
  if (!igl::is_edge_manifold(F)||!igl::is_vertex_manifold(F, BI)) 
  {
    std::cout<<"Not Manifold"<<std::endl;
    return false;
  }

  Polyhedron mesh;
  if (!igl::copyleft::cgal::mesh_to_polyhedron(V,F,mesh)) 
  {
    std::cerr << "Convert Failure." << std::endl;
    return false;
  }

  typedef std::map<Polyhedron::Facet_const_handle, double> Facet_double_map;
  Facet_double_map internal_sdf_map;
  boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);
  // compute SDF values using default parameters for number of rays, and cone angle
  CGAL::sdf_values(mesh, sdf_property_map);
  // create a property-map for segment-ids
  typedef std::map<Polyhedron::Facet_const_handle, std::size_t> Facet_int_map;
  Facet_int_map internal_segment_map;
  boost::associative_property_map<Facet_int_map> segment_property_map(internal_segment_map);
  // segment the mesh using default parameters for number of levels, and smoothing lambda
  // Any other scalar values can be used instead of using SDF values computed using the CGAL function
  std::size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map);
  // print segment-ids
  std::vector<int> seg_ids;
  for(Polyhedron::Facet_const_iterator facet_it = mesh.facets_begin();
      facet_it != mesh.facets_end(); ++facet_it) {
      seg_ids.push_back( segment_property_map[facet_it]);
  }
  std::cout << std::endl;
  // const std::size_t number_of_clusters = 4;       // use 4 clusters in soft clustering
  // const double smoothing_lambda = 0.3;  // importance of surface features, suggested to be in-between [0,1]
  // // Note that we can use the same SDF values (sdf_property_map) over and over again for segmentation.
  // // This feature is relevant for segmenting the mesh several times with different parameters.
  // CGAL::segmentation_from_sdf_values(
  //   mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);


  C.resize(seg_ids.size(), 1);
  for (int i =0; i<seg_ids.size(); i++) {
    C(i) = seg_ids[i];
  }
  return true;
}

  void log_normalize_probability_matrix(std::vector<std::vector<double> >&
                                        probabilities) {
    const double epsilon = 5e-6;
    for(std::vector<std::vector<double> >::iterator it_i = probabilities.begin();
        it_i != probabilities.end(); ++it_i) {
      for(std::vector<double>::iterator it = it_i->begin(); it != it_i->end(); ++it) {
        double probability = (std::max)(*it,
                                        epsilon); // give every facet a little probability to be in any cluster
        probability = -log(probability);
        *it = (std::max)(probability, std::numeric_limits<double>::epsilon());
        // zero values are not accepted in max-flow as weights for edges which connects some vertex with Source or Sink (in boost::boykov..)
      }
    }
  }

 void calculate_and_log_normalize_dihedral_angles(
      const Eigen::MatrixXd &V, const Eigen::MatrixXi& F,
      double smoothing_lambda,
      std::vector<std::pair<std::size_t, std::size_t> >& a_edges,
      std::vector<double>& edge_weights) 
      {
        const double epsilon = 5e-6;
        Eigen::MatrixXi TT;
        Eigen::MatrixXd FN;

        igl::triangle_triangle_adjacency(F,TT);
        igl::per_face_normals(V,F,FN);

        auto angle_between = [](const Eigen::MatrixXd& FN, int i, int o){
          double inner = std::min(std::max(FN.row(i).dot(FN.row(o)),-1+1e-9), 1-1e-9);
          double angle = acos(inner)/igl::PI;
          return 1 - abs(angle);
        };
        a_edges.clear();
        for (int i=0; i<TT.rows(); i++) {
          for (int j=0; j<3; j++) {
              if (TT(i,j) == -1) continue;
              int of = TT(i,j);
              a_edges.push_back(std::make_pair<std::size_t, std::size_t>(i, of));
              double angle = angle_between(FN, i, of);
                      
              angle = (std::max)(angle, epsilon);
              angle = -log(angle);
              angle *= smoothing_lambda;

              edge_weights.push_back(angle);
          }
        }
  }

int graphcut_from_cgal(
      const Eigen::MatrixXd &V, const Eigen::MatrixXi& F,
   std::vector<std::vector<double>> &probability_matrix,
                      std::vector<std::size_t>& labels,
                      float smoothing_lambda=0.26){
    // calculating edge weights
    std::vector<std::pair<std::size_t, std::size_t> > edges;
    std::vector<double> edge_weights;
    log_normalize_probability_matrix(probability_matrix);
    calculate_and_log_normalize_dihedral_angles(V,F, smoothing_lambda, edges,
        edge_weights);  

    for(auto v: edge_weights){
      if (!std::isfinite(v)) {
        std::cout<<"Some NaN"<<std::endl;
        return 0;
      }
    }
        for(auto p: probability_matrix){
          for(auto v: p)
      if (!std::isfinite(v)) {
        std::cout<<"Some NaN"<<std::endl;
        return 0;
      }
    }
    CGAL::internal::Alpha_expansion_graph_cut_boykov_kolmogorov()(edges, edge_weights, probability_matrix, labels);
  }

void refine_labels_graph_cut(const Eigen::MatrixXd&V, const Eigen::MatrixXi&F,
                    const Eigen::MatrixXd& probability, Eigen::MatrixXi&C, 
                    float lambda=0.26){
    
    std::vector<std::vector<double>> prob_mat;
    igl::matrix_to_list(probability, prob_mat);
    std::vector<std::size_t> labels;
    for (int i=0; i<C.rows(); i++) labels.push_back(C(i));

    graphcut_from_cgal(V,F,prob_mat,labels, lambda);

    for (int i=0; i<C.rows(); i++) C(i) = labels[i];
}

namespace py = pybind11;
PYBIND11_MODULE(pygraphcut, m) {
    m.doc() = R"(as a test)";

    m.def("add_any", [](py::EigenDRef<Eigen::MatrixXd> x, int r, int c, double v) { x(r,c) += v; });
    m.def("refine_labels", [](const Eigen::MatrixXd&V, const Eigen::MatrixXi&F, const Eigen::MatrixXd& probability,
                              const Eigen::MatrixXi&C, float lambda=0.26, bool verbose=false) {
      Eigen::MatrixXi NC = C;
      refine_labels_graph_cut(V,F,probability, NC, lambda);
      return NC;
    });
    m.def("SDF_segment", [](const Eigen::MatrixXd&V, const Eigen::MatrixXi&F){
        Eigen::MatrixXd C;
        sdf_segmentation(V,F,C);
        return C.cast<int>();
    });
}
