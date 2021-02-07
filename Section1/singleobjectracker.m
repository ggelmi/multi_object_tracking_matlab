classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
            %       P_G: gating size in decimal --- scalar
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function estimates = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1   
            
            %Size of the measurements sequences
            N = size(Z,1);
            state_pred = state;
            
            estimates = cell(N,1) ;
            state_pred = state;
            
            for i = 1:N
            
             z = Z{i};
             
            [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(state_pred,z, measmodel, obj.gating.size);
            
             % The number of hypothesis
             mk = size(z_ingate,2) + 1;
             
            predicted_likelihood = exp(obj.density.predictedLikelihood(state_pred,z_ingate,measmodel));
             
             % weight of objected being detected
            weight_theta_k = (sensormodel.P_D * predicted_likelihood)/sensormodel.intensity_c;
            
            % Finding the max weight and measurement index
            
             [max_weight_theta,max_theta_Index] = max(weight_theta_k);
             
             %weight of object being missed
            weight_theta_0 = 1 - sensormodel.P_D;
            
            if mk==1 ||  weight_theta_0 > max_weight_theta
                
                state_update = state_pred;
               
            else
                % computing nearest measurement for KF update usage
                nearest_neighbor_meas = z_ingate(:,max_theta_Index);
                state_update = obj.density.update(state_pred, nearest_neighbor_meas, measmodel); 
            end
            
            estimates{i} = state_update;
              
            state_pred = obj.density.predict(state_update, motionmodel);

            end
        end
        
        
        function estimates = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1  
            N = size(Z,1);
            state_pred = state;
            
            estimates = cell(N,1) ;
            p_prev = state;
            w_prev = 0;
            
            for i = 1:N
            
             z = Z{i,1};
             % erase new posterior and weight
             w_new = [];
             p_new = struct('x',{},'P',{});
             
             % size of previous hypothesis
             mk_prev = size(w_prev,1)
             
             for i_prev = 1:mk_prev
               % gating the measurements
            [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(p_prev(i_prev), z, measmodel, obj.gating.size);
             % The number of hypothesis
             mk_new = size(z_ingate,2) + 1;
             % creating object detection hypothesis for each detection
             % inside the gate
             for i_new = 1:mk_new-1
                 
                 predicted_likelihood_log = obj.density.predictedLikelihood(p_prev(i_prev),z_ingate(:,i_new),measmodel);
                 
                 w_new (end+1,1) =  w_prev (i_prev) + predicted_likelihood_log + log(sensormodel.P_D/sensormodel.intensity_c);
                 p_new(end+1,1) =   obj.density.update(p_prev(i_prev), z_ingate(:,i_new), measmodel);
                 
             end
             % Creating missed detection hypothesis for each hypothesis
             
              w_new (end+1,1) = w_prev (i_prev) + log(1 - sensormodel.P_D);
              
              p_new(end+1,1) = p_prev(i_prev);
              
             end
             
             % Normalizing the weights
             
              w_new  = normalizeLogWeights(w_new);
              
              % pruning the hypothesis with smaller weights
              
              [w_new, p_new] = hypothesisReduction.prune(w_new, p_new, obj.reduction.w_min);
              
              % renormalizing the weights
              
              w_new  = normalizeLogWeights(w_new);
              
              % Merging different hypothesis
              
              [w_new, p_new] = hypothesisReduction.merge(w_new, p_new, obj.reduction.merging_threshold,obj.density);
              
              % capping the number of hypothesis and then renomalize the
              % weights
              [w_new,p_new] = hypothesisReduction.cap( w_new, p_new, 1 );
              % renormalizing the weights
              
              w_new  = normalizeLogWeights(w_new); % Not neccesary since we have only one hypothesis. But we can 
              % keep more than one hypothesis, we have to the normalization
              
              % extracting the object estimate with highest weight
              [best_width,best_w_idx] = max(w_new);
              
              estimates{i} = p_new(best_w_idx);
              
              % updating
              p_prev = p_new;
              w_prev = w_new;
              
              % predicting the next state for each hypothesis
              
              m = length(w_prev);
              
              for i_oldhypo = 1:m
                  
                  p_prev(i_oldhypo) = obj.density.predict(p_prev(i_oldhypo), motionmodel);
                  
              end
              
            end
            
        end
        
        function estimates = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1  
            
             N = size(Z,1);
            state_pred = state;
            
            estimates = cell(N,1) ;
            
            p_prev = state;
            w_prev = 0;
            
            for i = 1:N
             z = Z{i,1};
             % erase new posterior and weight
             w_new = [];
             p_new = struct('x',{},'P',{});
             
             % size of previous hypothesis
             mk_prev = size(w_prev,1); 
               
             for i_prev = 1:mk_prev
               % gating the measurements
            [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(p_prev(i_prev), z, measmodel, obj.gating.size);
             % The number of hypothesis
             mk_new = size(z_ingate,2) + 1;
             
             % creating object detection hypothesis for each detection
             % inside the gate
             for i_new = 1:mk_new-1
                 
                 predicted_likelihood_log = obj.density.predictedLikelihood(p_prev(i_prev),z_ingate(:,i_new),measmodel);
                 
                 w_new (end+1,1) =  w_prev (i_prev) + predicted_likelihood_log + log(sensormodel.P_D/sensormodel.intensity_c);
                 p_new(end+1,1) =   obj.density.update(p_prev(i_prev), z_ingate(:,i_new), measmodel);
                 
             end
             % Creating missed detection hypothesis for each hypothesis
             
              w_new (end+1,1) = w_prev (i_prev) + log(1 - sensormodel.P_D);
              
              p_new(end+1,1) = p_prev(i_prev);
             end
                
             % Normalizing the weights
             
              w_new  = normalizeLogWeights(w_new);
              
              % pruning the hypothesis with smaller weights
              
              [w_new, p_new] = hypothesisReduction.prune(w_new, p_new, obj.reduction.w_min);
              
              % renormalizing the weights
              
              w_new  = normalizeLogWeights(w_new);
              
              % Merging different hypothesis
              
              [w_new, p_new] = hypothesisReduction.merge(w_new, p_new, obj.reduction.merging_threshold,obj.density);
              
              % capping the number of hypothesis and then renomalize the
              % weights
              [w_new,p_new] = hypothesisReduction.cap( w_new, p_new, obj.reduction.M );
              % renormalizing the weights
              
              w_new  = normalizeLogWeights(w_new);
             % extracting the object estimate with highest weight
              [best_width,best_w_idx] = max(w_new);
              
              estimates{i} = p_new(best_w_idx);
              
              % updating
              p_prev = p_new;
              w_prev = w_new;
              
              % predicting the next state for each hypothesis
              
              m = length(w_prev);
              
              for i_oldhypo = 1:m
                  
                  p_prev(i_oldhypo) = obj.density.predict(p_prev(i_oldhypo), motionmodel);
                  
              end
            end

        end
        
    end
end

