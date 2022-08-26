from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
import scipy.spatial.distance as dist
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return dist.cdist(pointA, documents).reshape(-1, 1)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:    
    graph = {}
    
    for i in range(len(data)):
        distances = dist_f(data[i].reshape(1,-1),
                           data)
        distances = distances.reshape(-1)
                
        order_short = distances.argsort()
        order_short = order_short[order_short != i]
        idx_short_select = np.random.choice(num_candidates_for_choice_short,
                                      size=num_edges_short,
                                      replace=False)
        
        graph[i] = order_short[idx_short_select].tolist()
        
        order_long = distances.argsort()[::-1]
        order_long = order_long[order_long != i]
        idx_long_select = np.random.choice(num_candidates_for_choice_long,
                                      size=num_edges_long,
                                      replace=False)
        
        graph[i] += order_long[idx_long_select].tolist()
    
    return graph
    

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    documents = []
        
    all_edges = np.array(list(graph_edges.keys()))
    
    queue_flag = False
    queue_edges = list()
    queue_distances = list()
    candidates_edges = list()
    candidates_distances = list()
    explored_edges = list()
    distance_min = 1e12
    parent_idx = None
    idx = np.random.choice(all_edges,
                           size=1,
                           replace=False)[0]
    while len(documents) != search_k:
        edges = np.array(graph_edges[idx])
        edges = edges[~np.isin(edges, explored_edges)]
        
        distances = dist_f(query_point.reshape(1, -1),
                           all_documents[edges])
        distances = distances.reshape(-1)
        
        print('-'*50)
        print(f'idx {idx}',
              f'distance_min {distance_min}',
              f'explored_edges {explored_edges}',
              f'queue_edges {queue_edges}',
              f'queue_distances {queue_distances}',
              f'candidates_edges {candidates_edges}',
              f'candidates_distances {candidates_distances}',
              f'dicuments {documents}',
              f'all_edges {graph_edges[idx]}',
              f'filtered_edges {edges}',
              f'distances {distances}',
              f'current_min {distances.min()}',
              f'current_min_idx {edges[distances.argmin()]}',
              sep='\n')
        
        if (distances.min() > distance_min
            and idx not in documents):
            if not queue_flag:
                print('Ini queue')
                queue_flag = True
    
#                 if parent_idx:
#                     edges = np.array(graph_edges[parent_idx])
#                     edges = edges[~np.isin(edges, explored_edges)]

#                     distances = dist_f(query_point.reshape(1, -1),
#                                        all_documents[edges])
#                     distances = distances.reshape(-1)
                    
                queue_edges = list(edges[distances.argsort()])[1:]
                queue_distances = list(np.sort(distances))[1:]
                candidates_edges = list()
                candidates_distances = list()
            explored_edges.append(idx) 
            
            if len(queue_edges) != 0:
                print('Pop queue')
                candidates_edges.append(idx)
                candidates_distances.append(distance_min)
                
                idx = queue_edges[0]
                distance_min = queue_distances[0]
                queue_edges.remove(queue_edges[0])
                queue_distances.remove(queue_distances[0])
            else:
                print('Append')
                queue_flag = False
                documents.append(candidates_edges[np.argmin(candidates_distances)])
                explored_edges = list(documents)
                distance_min = 1e12
                parent_idx = None
                idx = np.random.choice(all_edges[~np.isin(all_edges,
                                                          explored_edges)],
                                       size=1,
                                       replace=False)[0]
        elif (distances.min() > distance_min
              and idx in documents):
            explored_edges.append(idx)     
#             print('Restart')            
            distance_min = 1e12
            idx = np.random.choice(all_edges[~np.isin(all_edges,
                                                      explored_edges)],
                                   size=1,
                                   replace=False)[0]
        else:
            explored_edges.append(idx)     
            parent_idx = idx
            idx = edges[distances.argmin()]
            distance_min = distances.min()   
    return np.array(documents)

