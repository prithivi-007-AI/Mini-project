import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from dateutil.parser import parse
from collections import defaultdict
import textstat
import pickle
import os

class StudyPlanner:
    def __init__(self):
        self.topics = []
        self.time_remaining = None
        self.schedule = []
        self.todo_list = []
        self.completed_topics = []
        self.complexity_model = textstat
        
    def add_topic(self, topic, hours_required=None):
        self.topics.append({
            'name': topic,
            'hours_required': hours_required,
            'complexity': None,
            'priority': None
        })
    
    def set_time_remaining(self, days, hours_per_day):
        self.time_remaining = {
            'total_days': days,
            'hours_per_day': hours_per_day,
            'total_hours': days * hours_per_day
        }
    
    def analyze_complexity(self):
        for topic in self.topics:
            readability = self.complexity_model.flesch_reading_ease(topic['name'])
            complexity = 1 - (readability / 100) if readability > 0 else 0.5
            complexity = max(0.1, min(0.9, complexity))
            topic['complexity'] = complexity
            if topic['hours_required'] is None:
                topic['hours_required'] = max(1, round(complexity * 5))
    
    def prioritize_topics(self):
        max_complexity = max(t['complexity'] for t in self.topics)
        max_hours = max(t['hours_required'] for t in self.topics)
        for topic in self.topics:
            norm_complexity = topic['complexity'] / max_complexity
            norm_hours = topic['hours_required'] / max_hours
            topic['priority'] = (0.7 * norm_complexity) + (0.3 * (1 - norm_hours))
        self.topics.sort(key=lambda x: x['priority'], reverse=True)
    
    def create_schedule(self):
        if not self.time_remaining:
            raise ValueError("Time remaining not set. Call set_time_remaining() first.")
        
        self.analyze_complexity()
        self.prioritize_topics()
        
        total_hours_available = self.time_remaining['total_hours']
        total_hours_required = sum(t['hours_required'] for t in self.topics)
        
        if total_hours_required > total_hours_available:
            print(f"Warning: You need {total_hours_required} hours but only have {total_hours_available} hours.")
            scale_factor = total_hours_available / total_hours_required
            for topic in self.topics:
                topic['hours_required'] = max(1, round(topic['hours_required'] * scale_factor))
        
        current_date = datetime.now().date()
        daily_hours_remaining = self.time_remaining['hours_per_day']
        day_counter = 0
        daily_schedule = defaultdict(list)
        
        for topic in self.topics:
            hours_left = topic['hours_required']
            while hours_left > 0 and day_counter < self.time_remaining['total_days']:
                date_key = current_date + timedelta(days=day_counter)
                time_allocated = min(hours_left, daily_hours_remaining)
                daily_schedule[date_key].append({
                    'topic': topic['name'],
                    'hours': time_allocated,
                    'completed': False
                })
                hours_left -= time_allocated
                daily_hours_remaining -= time_allocated
                if daily_hours_remaining <= 0:
                    day_counter += 1
                    daily_hours_remaining = self.time_remaining['hours_per_day']
        self.schedule = sorted(daily_schedule.items())
    
    def generate_todo_list(self):
        self.todo_list = []
        for date, tasks in self.schedule:
            for task in tasks:
                self.todo_list.append({
                    'date': date,
                    'task': f"Study {task['topic']} for {task['hours']} hours",
                    'completed': task['completed']
                })
        return self.todo_list
    
    def mark_completed(self, task_description):
        for item in self.todo_list:
            if item['task'] == task_description and not item['completed']:
                item['completed'] = True
                for date, tasks in self.schedule:
                    for task in tasks:
                        if task['topic'] in item['task'] and not task['completed']:
                            task['completed'] = True
                            self.completed_topics.append(task['topic'])
                return True
        return False
    
    def get_progress(self):
        total_tasks = len(self.todo_list)
        completed_tasks = sum(1 for t in self.todo_list if t['completed'])
        return {
            'completed': completed_tasks,
            'total': total_tasks,
            'percentage': (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        }
    
    def save_progress(self, filename="study_planner.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump({
                'topics': self.topics,
                'time_remaining': self.time_remaining,
                'schedule': self.schedule,
                'todo_list': self.todo_list,
                'completed_topics': self.completed_topics
            }, f)
        print(f"Progress saved to {filename}")
    
    def load_progress(self, filename="study_planner.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.topics = data['topics']
                self.time_remaining = data['time_remaining']
                self.schedule = data['schedule']
                self.todo_list = data['todo_list']
                self.completed_topics = data['completed_topics']
            print(f"Progress loaded from {filename}")
            return True
        return False

class StudyPlannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Study Planner")
        self.root.geometry("900x650")
        self.root.minsize(800, 600)
        
        # Custom style for a modern look
        self.setup_styles()
        
        # Initialize planner
        self.planner = StudyPlanner()
        
        # Create main container
        self.main_container = ttk.Frame(root, style='Main.TFrame')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.header = ttk.Frame(self.main_container, style='Header.TFrame')
        self.header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(self.header, text="AI Study Planner", style='Header.TLabel').pack(side=tk.LEFT)
        
        # Create notebook for multi-page interface
        self.notebook = ttk.Notebook(self.main_container, style='TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create frames
        self.create_topic_frame()
        self.create_time_frame()
        self.create_schedule_frame()
        self.create_todo_frame()
        self.create_progress_frame()
        
        # Disable all tabs except the first one initially
        self.disable_tabs([1, 2, 3, 4])
        
        # Try to load saved progress
        self.load_progress()
    
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        
        # Main background
        style.configure('Main.TFrame', background='#f0f0f0')
        
        # Header
        style.configure('Header.TFrame', background='#4b6cb7')
        style.configure('Header.TLabel', 
                        background='#4b6cb7', 
                        foreground='white', 
                        font=('Helvetica', 16, 'bold'),
                        padding=10)
        
        # Notebook style
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', 
                        font=('Helvetica', 10, 'bold'),
                        padding=[10, 5])
        style.map('TNotebook.Tab',
                 background=[('selected', '#4b6cb7'), ('!selected', '#e0e0e0')],
                 foreground=[('selected', 'white'), ('!selected', 'black')])
        
        # Button styles
        style.configure('Accent.TButton', 
                       font=('Helvetica', 10, 'bold'),
                       background='#4b6cb7',
                       foreground='white',
                       borderwidth=1,
                       focusthickness=3,
                       focuscolor='none')
        style.map('Accent.TButton',
                 background=[('active', '#3a56a0'), ('!active', '#4b6cb7')],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        # Entry styles
        style.configure('TEntry', 
                       font=('Helvetica', 11),
                       padding=5)
        
        # Listbox styles
        style.configure('TListbox', 
                       font=('Helvetica', 11),
                       background='white',
                       foreground='black')
        
        # Treeview styles
        style.configure('Treeview',
                       font=('Helvetica', 11),
                       rowheight=25)
        style.configure('Treeview.Heading',
                      font=('Helvetica', 12, 'bold'))
        
        # Progress bar
        style.configure('Custom.Horizontal.TProgressbar',
                      thickness=20,
                      troughcolor='#e0e0e0',
                      background='#4b6cb7',
                      lightcolor='#6d8ad6',
                      darkcolor='#3a56a0')
    
    def create_topic_frame(self):
        """Create the topic entry frame"""
        self.topic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.topic_frame, text="1. Topics")
        
        # Content frame with padding
        content_frame = ttk.Frame(self.topic_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, 
                 text="Enter Your Study Topics", 
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 15))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(content_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.topic_listbox = tk.Listbox(list_frame, 
                                      font=('Helvetica', 12),
                                      selectbackground='#4b6cb7',
                                      selectforeground='white')
        self.topic_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.topic_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.topic_listbox.config(yscrollcommand=scrollbar.set)
        
        # Entry fields
        entry_frame = ttk.Frame(content_frame)
        entry_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(entry_frame, text="Topic:", font=('Helvetica', 11)).grid(row=0, column=0, padx=5, sticky=tk.E)
        self.topic_entry = ttk.Entry(entry_frame, font=('Helvetica', 11))
        self.topic_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        
        ttk.Label(entry_frame, text="Hours (optional):", font=('Helvetica', 11)).grid(row=0, column=2, padx=5)
        self.hours_entry = ttk.Entry(entry_frame, font=('Helvetica', 11), width=8)
        self.hours_entry.grid(row=0, column=3, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, 
                  text="Add Topic", 
                  style='Accent.TButton',
                  command=self.add_topic).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="Remove Selected", 
                  command=self.remove_topic).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="Next →", 
                  style='Accent.TButton',
                  command=self.next_to_time).pack(side=tk.RIGHT, padx=5)
        
        # Configure grid weights
        entry_frame.columnconfigure(1, weight=1)
    
    def create_time_frame(self):
        """Create the time entry frame"""
        self.time_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_frame, text="2. Time")
        
        # Content frame with padding
        content_frame = ttk.Frame(self.time_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, 
                 text="Set Your Study Time", 
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 20))
        
        # Time entry fields
        time_entry_frame = ttk.Frame(content_frame)
        time_entry_frame.pack(pady=20)
        
        ttk.Label(time_entry_frame, 
                 text="Days until deadline:", 
                 font=('Helvetica', 11)).grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)
        self.days_entry = ttk.Entry(time_entry_frame, font=('Helvetica', 11))
        self.days_entry.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        
        ttk.Label(time_entry_frame, 
                 text="Hours per day:", 
                 font=('Helvetica', 11)).grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
        self.hours_per_day_entry = ttk.Entry(time_entry_frame, font=('Helvetica', 11))
        self.hours_per_day_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, 
                  text="← Back", 
                  command=self.back_to_topics).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="Generate Schedule →", 
                  style='Accent.TButton',
                  command=self.generate_schedule).pack(side=tk.RIGHT, padx=5)
    
    def create_schedule_frame(self):
        """Create the schedule display frame"""
        self.schedule_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.schedule_frame, text="3. Schedule")
        
        # Content frame with padding
        content_frame = ttk.Frame(self.schedule_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, 
                 text="Your Study Schedule", 
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 15))
        
        # Treeview with scrollbars
        tree_frame = ttk.Frame(content_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.schedule_tree = ttk.Treeview(tree_frame, columns=('Date', 'Topic', 'Hours'), show='headings')
        self.schedule_tree.heading('Date', text='Date')
        self.schedule_tree.heading('Topic', text='Topic')
        self.schedule_tree.heading('Hours', text='Hours')
        self.schedule_tree.column('Date', width=150, anchor=tk.W)
        self.schedule_tree.column('Topic', width=400, anchor=tk.W)
        self.schedule_tree.column('Hours', width=80, anchor=tk.CENTER)
        
        y_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.schedule_tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_tree.config(yscrollcommand=y_scrollbar.set)
        
        x_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.schedule_tree.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.schedule_tree.config(xscrollcommand=x_scrollbar.set)
        
        self.schedule_tree.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(button_frame, 
                  text="← Back", 
                  command=self.back_to_time).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="Generate To-Do List →", 
                  style='Accent.TButton',
                  command=self.ask_generate_todo).pack(side=tk.RIGHT, padx=5)
    
    def create_todo_frame(self):
        """Create the to-do list frame"""
        self.todo_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.todo_frame, text="4. To-Do List")
        
        # Content frame with padding
        content_frame = ttk.Frame(self.todo_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, 
                 text="Your Study To-Do List", 
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 15))
        
        # Treeview with scrollbars
        tree_frame = ttk.Frame(content_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.todo_tree = ttk.Treeview(tree_frame, columns=('Date', 'Task', 'Completed'), show='headings')
        self.todo_tree.heading('Date', text='Date')
        self.todo_tree.heading('Task', text='Task')
        self.todo_tree.heading('Completed', text='Completed')
        self.todo_tree.column('Date', width=120, anchor=tk.W)
        self.todo_tree.column('Task', width=450, anchor=tk.W)
        self.todo_tree.column('Completed', width=80, anchor=tk.CENTER)
        
        y_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.todo_tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.todo_tree.config(yscrollcommand=y_scrollbar.set)
        
        x_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.todo_tree.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.todo_tree.config(xscrollcommand=x_scrollbar.set)
        
        self.todo_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind double click to mark as complete
        self.todo_tree.bind("<Double-1>", self.toggle_task_completion)
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(button_frame, 
                  text="← Back", 
                  command=self.back_to_schedule).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="View Progress →", 
                  style='Accent.TButton',
                  command=self.show_progress).pack(side=tk.RIGHT, padx=5)
    
    def create_progress_frame(self):
        """Create the progress tracking frame"""
        self.progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_frame, text="5. Progress")
        
        # Content frame with padding
        content_frame = ttk.Frame(self.progress_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(content_frame, 
                 text="Your Study Progress", 
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 15))
        
        # Progress summary
        progress_summary = ttk.Frame(content_frame)
        progress_summary.pack(fill=tk.X, pady=(0, 20))
        
        self.progress_label = ttk.Label(progress_summary, 
                                      text="Completed: 0/0 tasks (0%)", 
                                      font=('Helvetica', 12))
        self.progress_label.pack(side=tk.TOP, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_summary, 
                                          style='Custom.Horizontal.TProgressbar',
                                          orient=tk.HORIZONTAL, 
                                          length=600)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Completed topics
        completed_frame = ttk.LabelFrame(content_frame, 
                                       text="Completed Topics",
                                       padding=10)
        completed_frame.pack(fill=tk.BOTH, expand=True)
        
        self.completed_listbox = tk.Listbox(completed_frame, 
                                         font=('Helvetica', 11),
                                         selectbackground='#4b6cb7',
                                         selectforeground='white')
        self.completed_listbox.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(completed_frame, orient="vertical", command=self.completed_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.completed_listbox.config(yscrollcommand=scrollbar.set)
        
        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(button_frame, 
                  text="← Back", 
                  command=self.back_to_todo).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                  text="Save Progress", 
                  style='Accent.TButton',
                  command=self.save_progress).pack(side=tk.RIGHT, padx=5)
    
    def disable_tabs(self, tab_indices):
        """Disable the specified tabs (0-based index)"""
        for i in tab_indices:
            self.notebook.tab(i, state="disabled")
    
    def enable_tab(self, tab_index):
        """Enable the specified tab (0-based index)"""
        self.notebook.tab(tab_index, state="normal")
    
    def add_topic(self):
        """Add a topic to the list"""
        topic = self.topic_entry.get().strip()
        if not topic:
            messagebox.showwarning("Warning", "Please enter a topic name.")
            return
            
        hours = self.hours_entry.get().strip()
        try:
            hours = int(hours) if hours else None
        except ValueError:
            messagebox.showwarning("Warning", "Hours must be a number. Using auto-estimate instead.")
            hours = None
            
        self.planner.add_topic(topic, hours)
        self.topic_listbox.insert(tk.END, f"{topic}" + (f" ({hours} hours)" if hours else ""))
        self.topic_entry.delete(0, tk.END)
        self.hours_entry.delete(0, tk.END)
        
    def remove_topic(self):
        """Remove selected topic from the list"""
        selection = self.topic_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a topic to remove.")
            return
            
        index = selection[0]
        self.topic_listbox.delete(index)
        del self.planner.topics[index]
        
    def next_to_time(self):
        """Move to the time entry frame"""
        if not self.planner.topics:
            messagebox.showwarning("Warning", "Please add at least one topic.")
            return
            
        self.enable_tab(1)
        self.notebook.select(1)
        
    def back_to_topics(self):
        """Move back to the topic entry frame"""
        self.notebook.select(0)
        
    def generate_schedule(self):
        """Generate the study schedule"""
        try:
            days = int(self.days_entry.get())
            hours_per_day = int(self.hours_per_day_entry.get())
            if days <= 0 or hours_per_day <= 0:
                raise ValueError("Numbers must be positive.")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid positive numbers for days and hours.")
            return
            
        self.planner.set_time_remaining(days, hours_per_day)
        self.planner.create_schedule()
        
        # Display schedule in the treeview
        self.schedule_tree.delete(*self.schedule_tree.get_children())
        for date, tasks in self.planner.schedule:
            for task in tasks:
                self.schedule_tree.insert('', tk.END, values=(
                    date.strftime('%Y-%m-%d, %A'),
                    task['topic'],
                    task['hours']
                ))
        
        self.enable_tab(2)
        self.notebook.select(2)
        
    def back_to_time(self):
        """Move back to the time entry frame"""
        self.notebook.select(1)
        
    def ask_generate_todo(self):
        """Ask user if they want to generate a to-do list"""
        if messagebox.askyesno("Generate To-Do List", "Would you like to generate a to-do list from this schedule?"):
            self.generate_todo_list()
            
    def generate_todo_list(self):
        """Generate and display the to-do list"""
        self.planner.generate_todo_list()
        
        # Display to-do list in the treeview
        self.todo_tree.delete(*self.todo_tree.get_children())
        for item in self.planner.todo_list:
            self.todo_tree.insert('', tk.END, values=(
                item['date'].strftime('%Y-%m-%d'),
                item['task'],
                "✓" if item['completed'] else "◻"
            ))
        
        self.enable_tab(3)
        self.notebook.select(3)
        
    def toggle_task_completion(self, event):
        """Toggle task completion status when double-clicked"""
        item = self.todo_tree.selection()[0]
        values = self.todo_tree.item(item, 'values')
        
        if values[2] == "◻":
            # Mark as completed
            if self.planner.mark_completed(values[1]):
                self.todo_tree.item(item, values=(values[0], values[1], "✓"))
                self.update_progress()
            else:
                messagebox.showwarning("Warning", "Could not mark task as completed.")
        else:
            # Already completed - unmark if needed
            if messagebox.askyesno("Unmark Task", "This task is already completed. Unmark it?"):
                # This would require adding an unmark_completed method to your StudyPlanner class
                messagebox.showinfo("Info", "This feature would require additional implementation.")
                
    def show_progress(self):
        """Show progress tracking"""
        self.update_progress()
        self.notebook.select(4)
        
    def update_progress(self):
        """Update the progress display"""
        progress = self.planner.get_progress()
        
        self.progress_label.config(
            text=f"Completed: {progress['completed']}/{progress['total']} tasks ({progress['percentage']:.1f}%)"
        )
        self.progress_bar['value'] = progress['percentage']
        
        # Update completed topics list
        self.completed_listbox.delete(0, tk.END)
        for topic in self.planner.completed_topics:
            self.completed_listbox.insert(tk.END, topic)
        
    def back_to_todo(self):
        """Move back to the to-do list frame"""
        self.notebook.select(3)
        
    def back_to_schedule(self):
        """Move back to the schedule frame"""
        self.notebook.select(2)
        
    def save_progress(self):
        """Save the current progress"""
        self.planner.save_progress()
        messagebox.showinfo("Success", "Your progress has been saved successfully.")
    
    def load_progress(self):
        """Try to load saved progress"""
        if self.planner.load_progress():
            # Update UI with loaded data
            self.topic_listbox.delete(0, tk.END)
            for topic in self.planner.topics:
                self.topic_listbox.insert(tk.END, f"{topic['name']}" + 
                                        (f" ({topic['hours_required']} hours)" if topic['hours_required'] else ""))
            
            # Enable all tabs since we have data
            for i in range(1, 5):
                self.enable_tab(i)
            
            # If we have a schedule, show it
            if self.planner.schedule:
                self.schedule_tree.delete(*self.schedule_tree.get_children())
                for date, tasks in self.planner.schedule:
                    for task in tasks:
                        self.schedule_tree.insert('', tk.END, values=(
                            date.strftime('%Y-%m-%d, %A'),
                            task['topic'],
                            task['hours']
                        ))
            
            # If we have a to-do list, show it
            if self.planner.todo_list:
                self.todo_tree.delete(*self.todo_tree.get_children())
                for item in self.planner.todo_list:
                    self.todo_tree.insert('', tk.END, values=(
                        item['date'].strftime('%Y-%m-%d'),
                        item['task'],
                        "✓" if item['completed'] else "◻"
                    ))
            
            # Update progress
            self.update_progress()

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set window icon (replace with your own icon if available)
    try:
        root.iconbitmap('study_planner.ico')
    except:
        pass
    
    app = StudyPlannerApp(root)
    root.mainloop()
