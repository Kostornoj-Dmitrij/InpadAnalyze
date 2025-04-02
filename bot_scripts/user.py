class User:
    def __init__(self, user_id):
        self.user_id = ''
        self.state = 'chat_start'
        self.t_user_id = user_id
        self.callback = self
        self.tz_file_path = None
        self.result_file_path = None