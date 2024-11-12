    def check_split_screen(self, event):
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        if current_width < screen_width // 2 or current_height < screen_height // 2:
            self.show_split_screen_warning()

    def show_split_screen_warning(self):
        messagebox.showwarning("Split Screen Warning", "The application is in split screen mode!")