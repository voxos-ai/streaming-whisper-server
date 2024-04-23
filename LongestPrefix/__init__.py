class LongestPrefix:
    def __init__(self) -> None:
        self.commit_list:list = []
        self.moving_windown:list[list[str]] = []
        self.moving_windown_size:int = 10
    def flush(self):
        return " ".join(self.commit_list)
    def __add_to_moving_window(self,text:list[str]):
        if len(self.moving_windown) == 0:
            self.moving_windown.append(text)
        elif len(self.moving_windown) < self.moving_windown_size:
            self.moving_windown = [text] + self.moving_windown 
        else:
            self.moving_windown = [text] + self.moving_windown
            self.moving_windown.pop()
            self.make_commit()
            print(f"your data:  {self.flush()}")
        print("-------------------------------")
        for m in self.moving_windown:
            print(m)
        print("-------------------------------")

    def make_commit(self):
        assert(len(self.moving_windown) == self.moving_windown_size) ,"commit process only start after the buffer is filled"
        last_len = len(self.moving_windown[-1])
        
        for i in range(last_len):
            temp_flag = []
            match_word = self.moving_windown[-1][0]
            for j in range(1,self.moving_windown_size):
                if match_word == self.moving_windown[j][0]:
                    temp_flag.append(1)
                else:
                    temp_flag.append(0)
            if sum(temp_flag) == self.moving_windown_size - 1:
                self.commit_list.append(match_word)
                for k in range(self.moving_windown_size):
                    
                    self.moving_windown[k].pop(0)
            else:
                break

    
    def insert(self,text:str):
        text:list[str] = [i for i in text.split(" ") if i not in ["","!","?","|","]","[","{","}","/",">","<"]]
        print(text)
        if self.commit_list == []:
            self.__add_to_moving_window(text)
        else:
            print(self.commit_list)
            temp = []
            for idx_cmt in range(len(self.commit_list)):
                if text[idx_cmt] != self.commit_list[idx_cmt]:
                    temp.append(text[idx_cmt])
            temp += text[idx_cmt+1:]
            self.__add_to_moving_window(temp)
