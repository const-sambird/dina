import random

class WorkloadManager:
    def __init__(self, workload: list[str], templates: list[int], execution_mode: str, fraction: int | None):
        '''
        Container class for the workload to be sent to the Router.

        Note this is different from the (static) workload sent to the preprocessor,
        because we still want to generate a full complement of index candidates,
        but in a low-training-data environment or a workload shift environment,
        we need to modify which queries are actually sent to each DBMS instance
        for use in computing the reinforcement learning agent's reward function.

        :param workload: a list of every query in the full generated workload
        :param templates: which template # each query in the workload belongs to
        :param execution_mode: how should the workload change?
        :param fraction: what proportion of the full templates will be used in training
        '''
        self._workload = workload
        self._templates = templates
        self._partial_workload = workload
        self._partial_templates = templates
        self._full_workload = workload
        self._full_templates = templates
        self._exe_mode = execution_mode
        self._fraction = fraction

        if execution_mode == 'low_data':
            self.select_queries()

    def select_queries(self):
        '''
        In the low data and workload drift scenarioes, we need to
        select a fraction of the templates to be used in the training set.
        '''
        templates = list(set(self._full_templates))
        num_full_templates = len(templates)
        num_templates = round(num_full_templates * self._fraction)

        selected_templates = set()

        for _ in range(num_templates):
            template = random.choice(templates)
            selected_templates.add(template)
            templates.remove(template)
        
        self._partial_workload = []
        self._partial_templates = []

        for i, query in enumerate(self._full_workload):
            template = self._full_templates[i]
            if template in selected_templates:
                self._partial_workload.append(query)
                self._partial_templates.append(template)
        
        self._workload = self._partial_workload
        self._templates = self._partial_templates
    
    def update_workload(self):
        '''
        If we are in a workload drift experiment, then we need to vary which templates
        are present in the overall workload. If we are not in a workload drift experiment,
        then this function is a no-op.
        '''
        if self._exe_mode != 'drift':
            return
        
        self.select_queries()
    
    def workload(self) -> list[str]:
        '''
        Returns the training set workload to be used by the router.

        :returns: the queries in the training set
        '''
        return self._workload
    
    def templates(self) -> list[int]:
        '''
        Returns which template each query in the training set is generated from.

        :returns: the template assignment to each query
        '''
        return self._templates
    
    def set_to_partial(self):
        '''
        If we have set the currently active workload/template set to
        the full workload (ie to generate a routing table), we can
        reset it back to the partial one without reselecting templates here.
        '''
        self._workload = self._partial_workload
        self._templates = self._partial_templates
    
    def set_to_full(self):
        '''
        Changes the active workload to the full set, rather than the partial.
        '''
        self._workload = self._full_workload
        self._templates = self._full_templates
