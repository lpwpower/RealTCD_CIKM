class CDPrompt:
    def __init__(self, obj_task, add_cot = False, add_role = False, num_examplars = 1, cd_type = 1, args = None) -> None:
        # self.instructor_cd = f"A temporal causal graph is represented as a list of tuples, where each tuple (u, v, t) denotes that there is an causal edge from lag t of node u to comtemporal node v. For example, (6, 5, 2) denotes that node 6 is linked with node 5 at time 2. \n"
        if cd_type == 0:
            self.instructor_cd = f"A causal graph is a directed acyclic graph, represented as a list of tuples, where each tuple (u, v) denotes that there is an causal edge from node u to node v. For example, (6, 5) denotes that node 6 has a causal effect on node 5. \n"
        elif cd_type == 1:
            self.instructor_cd = f"A temporal causal graph is a directed acyclic graph, represented as a list of tuples, where each tuple (u, v, t) denotes that there is an temporal causal edge from t time lag version of node u to node v. For example, (6, 5, 2) denotes that 2 time lags version of node 6 has a causal effect on node 5. \n"
        else:
            raise NotImplementedError(f"cd_type {cd_type} not implemented")
        
        self.args = args
        if args:
            self.imp = self.args.__dict__.get("imp", 0)
        else:
            self.imp = 1
        # self.prompt_imp = get_imp(imp)
        self.prompt_cot = f"You can think it step by step.\n"
        self.add_cot = add_cot
        self.add_role = add_role
        self.num_examplars = num_examplars
        self.obj_task = obj_task
        self.cd_type = cd_type
        
    def generate_prompt_qa(self, context=None, query = None, answer = None, *args, **kwargs):
        # generate prompt components
        # instructor_role, instructor_cd = self.instructor_role if self.add_role else "", self.instructor_cd
        instructor_role = self.obj_task.generate_instructor_role() if self.add_role else ""
        instructor_cd = self.instructor_cd
        prompt_cot = self.prompt_cot if self.add_cot else ""
        
        # prompt_context = self.obj_task.generate_context_prompt(context)
        instructor_task = self.obj_task.generate_instructor_task()
        instructor_scence = self.obj_task.generate_instructor_scence()
        instructor_candi_nodes = self.obj_task.generate_instructor_candi_nodes(self.cd_type)
        instructor_location = self.obj_task.generate_instructor_location()
        
        prompt_imp = self.obj_task.generate_prompt_imp() if self.imp else ""
        instructor_answer = self.obj_task.generate_instructor_answer()
        # prompt_examplars = self.obj_task.generate_prompt_examplars(self.num_examplars) if self.num_examplars else ""
        # prompt_question = self.obj_task.generate_prompt_question(query)
        prompt_question = self.obj_task.generate_prompt_question()


        prompt_seq = [
            instructor_role,
            instructor_cd,
            instructor_task,
            instructor_scence,
            instructor_candi_nodes,
            instructor_location,
            # self.prompt_imp,
            prompt_imp,
            instructor_answer,
            # prompt_examplars,
            # prompt_context,
            prompt_question,
            prompt_cot
        ]
        
        prompt = "".join(prompt_seq)
        
        prompt_qa = {
            "prompt": prompt,
            "answer": answer,
        }
        return prompt_qa