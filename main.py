import csv
import pandas as pd
import numpy as np
import random
import sys
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from controllers import ask_dataset_selection, ask_logging_selection, ask_parameter_selection
from logger_utils import StdoutSilencer

class OutputLogger:
    """Class to handle output to file only (no terminal mirroring)"""
    def __init__(self, filename="jsp_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
        # Write header to log file
        self.log.write(f"Sailfish Job Shop Scheduling Optimizer Output Log\n")
        self.log.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*80}\n\n")
        self.log.flush()
    
    def write(self, message):
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.log.flush()
    
    def close(self):
        if self.log:
            self.log.close()

def read_jsp_data_from_csv(jobs_file="jobs.csv"):
    """
    Read Job Shop Scheduling data from CSV file.
    Expected format: Job, Operasi 1 Mesin, Operasi 1 Durasi, Operasi 2 Mesin, Operasi 2 Durasi, ...
    """
    try:
        jobs_df = pd.read_csv(jobs_file)
        jobs_data = []
        
        for _, row in jobs_df.iterrows():
            job_id = int(row['Job'])
            operations = []
            
            # Extract operations (assuming columns are: Job, Op1_Mesin, Op1_Durasi, Op2_Mesin, Op2_Durasi, ...)
            col_names = jobs_df.columns.tolist()
            op_cols = [col for col in col_names if 'Operasi' in col or 'Op' in col]
            
            # Group machine and duration columns
            for i in range(1, len(op_cols)//2 + 1):
                mesin_col = f'Operasi {i} Mesin' if f'Operasi {i} Mesin' in op_cols else f'Op{i}_Mesin'
                durasi_col = f'Operasi {i} Durasi' if f'Operasi {i} Durasi' in op_cols else f'Op{i}_Durasi'
                
                if mesin_col in row and durasi_col in row:
                    if pd.notna(row[mesin_col]) and pd.notna(row[durasi_col]):
                        operations.append({
                            'machine': int(row[mesin_col]),
                            'duration': int(row[durasi_col])
                        })
            
            jobs_data.append({
                'job_id': job_id,
                'operations': operations
            })
        
        return jobs_data
    
    except FileNotFoundError:
        print("CSV file not found. Using default Job Shop data.")
        return get_default_jsp_data()
    except pd.errors.EmptyDataError:
        print("CSV is empty or has no columns. Using default Job Shop data.")
        return get_default_jsp_data()

def get_default_jsp_data():
    """Return default Job Shop Scheduling data from the image"""
    jobs_data = [
        {
            'job_id': 1,
            'operations': [
                {'machine': 1, 'duration': 3},
                {'machine': 2, 'duration': 3},
                {'machine': 3, 'duration': 2}
            ]
        },
        {
            'job_id': 2,
            'operations': [
                {'machine': 1, 'duration': 1},
                {'machine': 3, 'duration': 5},
                {'machine': 2, 'duration': 3}
            ]
        },
        {
            'job_id': 3,
            'operations': [
                {'machine': 2, 'duration': 3},
                {'machine': 1, 'duration': 2},
                {'machine': 3, 'duration': 3}
            ]
        }
    ]
    
    return jobs_data

def create_job_operations_table(job_sequence, jobs_data, individual_name="Individual"):
    """
    Create a DataFrame table showing job operations and machines for the sequence
    """
    # Create the table data
    table_data = []
    max_ops = max(len(job['operations']) for job in jobs_data)
    
    # Create rows for Job, Operasi, Mesin, Durasi
    job_row = ['Job'] + job_sequence
    operasi_row = ['Operasi']
    mesin_row = ['Mesin']
    durasi_row = ['Durasi']
    
    # Track operation counters for each job
    job_counters = {job['job_id']: 0 for job in jobs_data}
    
    for job_id in job_sequence:
        # Find job data
        job_data = next(job for job in jobs_data if job['job_id'] == job_id)
        current_op_index = job_counters[job_id]
        
        if current_op_index < len(job_data['operations']):
            current_operation = job_data['operations'][current_op_index]
            operasi_row.append(current_op_index + 1)
            mesin_row.append(current_operation['machine'])
            durasi_row.append(current_operation['duration'])
            job_counters[job_id] += 1
        else:
            # Job completed, add empty cells
            operasi_row.append('-')
            mesin_row.append('-')
            durasi_row.append('-')
    
    # Create DataFrame
    df_data = {
        f'Pos{i}' if i > 0 else 'Attribute': [job_row[i], operasi_row[i], mesin_row[i], durasi_row[i]]
        for i in range(len(job_row))
    }
    
    df = pd.DataFrame(df_data, index=['Job', 'Operasi', 'Mesin', 'Durasi'])
    
    return df

def create_gantt_chart_visual(gantt_chart, makespan, individual_name="Individual"):
    """
    Create a Gantt chart visualization similar to the second image
    """
    if not gantt_chart:
        return None
    
    # Get all machines
    machines = sorted(set(entry['machine'] for entry in gantt_chart))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different jobs
    job_colors = {1: '#90EE90', 2: '#87CEEB', 3: '#FFA07A'}  # Light green, light blue, light salmon
    
    # Plot each operation
    for entry in gantt_chart:
        machine = entry['machine']
        start_time = entry['start_time']
        duration = entry['duration']
        job = entry['job']
        operation = entry['operation']
        
        # Machine position (reverse order to match the image)
        y_pos = len(machines) - machines.index(machine)
        
        # Create rectangle for the operation
        rect = patches.Rectangle(
            (start_time, y_pos - 0.4), 
            duration, 
            0.8, 
            linewidth=1, 
            edgecolor='black', 
            facecolor=job_colors.get(job, '#CCCCCC')
        )
        ax.add_patch(rect)
        
        # Add job and operation text
        ax.text(start_time + duration/2, y_pos, f'{job}{operation}{entry["machine"]}', 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Set up the plot
    ax.set_xlim(0, makespan + 1)
    ax.set_ylim(0.5, len(machines) + 0.5)
    
    # Set machine labels
    ax.set_yticks(range(1, len(machines) + 1))
    ax.set_yticklabels([f'Mesin {machines[len(machines)-i]}' for i in range(1, len(machines) + 1)])
    
    # Set time labels
    ax.set_xticks(range(0, makespan + 1))
    ax.set_xlabel('Makespan')
    ax.set_title(f'Gantt Chart - {individual_name} (Makespan: {makespan})')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def convert_random_to_schedule(random_values, jobs_data):
    """
    Convert random values to Job Shop Schedule using the specified method:
    1. Rank each position based on value order (smallest gets rank 1, etc.)
    2. Apply (rank-1 modulo n) + 1 transformation
    3. Create schedule based on job sequence
    
    Parameters:
    random_values: list of random values [0,1] with length m*n
    jobs_data: list of job information
    
    Returns:
    job_sequence: sequence of jobs for scheduling
    makespan: total makespan of the schedule
    gantt_chart: detailed scheduling information
    """
    n_jobs = len(jobs_data)
    n_positions = len(random_values)
    
    # Step 1: Create ranking - each position gets its rank in sorted value order
    value_index_pairs = [(random_values[i], i) for i in range(n_positions)]
    sorted_pairs = sorted(value_index_pairs)  # Sort by value
    
    # Create ranking: each original position gets its rank in sorted order
    position_ranks = [0] * n_positions
    for rank, (value, original_pos) in enumerate(sorted_pairs):
        position_ranks[original_pos] = rank + 1  # 1-based ranking
    
    # Step 2: Apply transformation (rank-1 modulo n) + 1
    job_sequence = []
    for rank in position_ranks:
        job_id = ((rank - 1) % n_jobs) + 1
        job_sequence.append(job_id)
    
    # Step 3: Calculate makespan and create schedule
    makespan, gantt_chart = calculate_makespan(job_sequence, jobs_data)
    
    return job_sequence, makespan, gantt_chart

def calculate_makespan(job_sequence, jobs_data, show_details=False):
    """
    Calculate makespan for given job sequence using Gantt chart method
    
    Parameters:
    job_sequence: sequence of jobs
    jobs_data: job operations data
    show_details: whether to show detailed calculation
    
    Returns:
    makespan: total makespan
    gantt_chart: scheduling details
    """
    # Initialize tracking structures
    job_counters = {job['job_id']: 0 for job in jobs_data}  # Track which operation each job is on
    machine_finish_times = {}  # Track when each machine becomes available
    job_finish_times = {job['job_id']: 0 for job in jobs_data}  # Track when each job's last operation finished
    
    gantt_chart = []
    
    if show_details:
        print(f"\nDetailed Job Shop Scheduling Calculation:")
        print("="*80)
        print(f"Job sequence: {job_sequence}")
        print()
    
    for step, job_id in enumerate(job_sequence):
        # Get current operation for this job
        current_op_index = job_counters[job_id]
        
        # Find job data
        job_data = next(job for job in jobs_data if job['job_id'] == job_id)
        
        # Check if job has more operations
        if current_op_index >= len(job_data['operations']):
            continue  # Skip if job is already complete
        
        current_operation = job_data['operations'][current_op_index]
        machine = current_operation['machine']
        duration = current_operation['duration']
        
        # Calculate start time
        machine_available = machine_finish_times.get(machine, 0)
        job_available = job_finish_times[job_id]
        start_time = max(machine_available, job_available)
        finish_time = start_time + duration
        
        # Update tracking structures
        machine_finish_times[machine] = finish_time
        job_finish_times[job_id] = finish_time
        job_counters[job_id] += 1
        
        # Record in Gantt chart
        gantt_entry = {
            'step': step + 1,
            'job': job_id,
            'operation': current_op_index + 1,
            'machine': machine,
            'duration': duration,
            'start_time': start_time,
            'finish_time': finish_time
        }
        gantt_chart.append(gantt_entry)
        
        if show_details:
            print(f"Step {step+1}: Job {job_id} Op{current_op_index+1} -> Machine {machine}")
            print(f"  Duration: {duration}, Start: {start_time}, Finish: {finish_time}")
    
    makespan = max(machine_finish_times.values()) if machine_finish_times else 0
    
    if show_details:
        print(f"\nMakespan: {makespan}")
        print("="*80)
    
    return makespan, gantt_chart

def print_jsp_data(jobs_data):
    """Print Job Shop Scheduling problem data"""
    print("\nJob Shop Scheduling Problem Data:")
    print("="*50)
    print(f"Number of Jobs: {len(jobs_data)}")
    
    max_operations = max(len(job['operations']) for job in jobs_data)
    print(f"Maximum Operations per Job: {max_operations}")
    
    machines = set()
    for job in jobs_data:
        for op in job['operations']:
            machines.add(op['machine'])
    print(f"Number of Machines: {len(machines)} (Machines: {sorted(machines)})")
    print()
    
    print("Job Details:")
    print(f"{'Job':<4} {'Operation':<10} {'Machine':<8} {'Duration':<8}")
    print("-" * 35)
    
    for job in jobs_data:
        for i, op in enumerate(job['operations']):
            print(f"{job['job_id']:<4} {i+1:<10} {op['machine']:<8} {op['duration']:<8}")
        if len(job['operations']) > 1:
            print("-" * 35)

class SailfishJSPOptimizer:
    def __init__(self, n_sailfish, n_sardines, jobs_data, max_iter=100, A=4, epsilon=0.001, log_to_file=True):
        """
        Initialize Sailfish Job Shop Scheduling Optimizer
        
        Parameters:
        n_sailfish: number of sailfish
        n_sardines: number of sardines
        jobs_data: list of job information dictionaries
        max_iter: maximum number of iterations
        A: sailfish optimizer parameter
        epsilon: convergence parameter
        log_to_file: whether to log output to file
        """
        if n_sardines <= n_sailfish:
            raise ValueError("Number of sardines must be greater than number of sailfish")
        
        self.original_n_sailfish = n_sailfish
        self.original_n_sardines = n_sardines
        self.n_sailfish = n_sailfish
        self.n_sardines = n_sardines
        self.jobs_data = jobs_data
        self.max_iter = max_iter
        self.A = A
        self.epsilon = epsilon
        
        # Calculate problem dimensions
        self.n_jobs = len(jobs_data)
        self.total_operations = sum(len(job['operations']) for job in jobs_data)
        self.problem_size = self.total_operations  # m*n where m=machines, n=jobs (approximated)
        
        # Set up logging
        self.log_to_file = log_to_file
        self.logger = None
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sailfish_jsp_SF{n_sailfish}_S{n_sardines}_{timestamp}.txt"
            self.logger = OutputLogger(filename)
            sys.stdout = self.logger
            print(f"JSP Output will be logged to: {filename}")
            print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Parameters: {n_sailfish} sailfish, {n_sardines} sardines, A={A}, epsilon={epsilon}")
            print("="*80 + "\n")
        
        # Population storage
        self.sailfish_random_values = []
        self.sailfish_sequences = []
        self.sailfish_fitness = []
        
        self.sardine_random_values = []
        self.sardine_sequences = []
        self.sardine_fitness = []
        
        # Original positions for updates
        self.original_sailfish_positions = []
        self.original_sardine_positions = []
        
        # Best solution tracking
        self.best_sequence = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        
        # Elite tracking
        self.elite_sailfish_fitness = None
        self.elite_sailfish_position = None
        self.injured_sardine_fitness = None
        self.injured_sardine_position = None
        
        # Algorithm variables
        self.lambda_k_values = []
        self.PD = None
        self.AP = None
        self.current_iteration = 0
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.log_to_file and self.logger:
            print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            sys.stdout = self.logger.terminal
            self.logger.close()
    
    def save_original_positions(self):
        """Save original positions before sorting and replacement"""
        self.original_sailfish_positions = [pos.copy() for pos in self.sailfish_random_values]
        self.original_sardine_positions = [pos.copy() for pos in self.sardine_random_values]
    
    def print_initial_parameters(self):
        """Print initial parameters and JSP data"""
        print("\n" + "="*80)
        print("1. INITIAL VARIABLES AND JSP DATA")
        print("="*80)
        
        print(f"Initial Parameters:")
        print(f"- Problem size: {self.n_jobs} jobs, {self.total_operations} total operations")
        print(f"- Sailfish population: {self.n_sailfish}")
        print(f"- Sardine population: {self.n_sardines}")
        print(f"- Maximum iterations: {self.max_iter}")
        print(f"- Parameter A: {self.A}")
        print(f"- Epsilon: {self.epsilon}")
        print()
        
        print_jsp_data(self.jobs_data)
    
    def generate_random_values(self, n_individuals):
        """Generate random values for each individual"""
        random_values = []
        for i in range(n_individuals):
            individual_values = [round(random.random(), 3) for _ in range(self.problem_size)]
            random_values.append(individual_values)
        return random_values
    
    def print_random_populations(self):
        """Print random values for sailfish and sardines"""
        print("\n" + "="*80)
        print("2. RANDOM SAILFISH AND SARDINES")
        print("="*80)
        
        self.sailfish_random_values = self.generate_random_values(self.n_sailfish)
        self.sardine_random_values = self.generate_random_values(self.n_sardines)
        
        print("SAILFISH Random Values:")
        print(f"{'ID':<8}", end="")
        for i in range(self.problem_size):
            print(f"P{i+1:2}", end="   ")
        print()
        
        for i in range(self.n_sailfish):
            print(f"SF{i+1:<6}", end="")
            for val in self.sailfish_random_values[i]:
                print(f"{val:5.3f}", end=" ")
            print()
        
        print("\nSARDINE Random Values:")
        print(f"{'ID':<8}", end="")
        for i in range(self.problem_size):
            print(f"P{i+1:2}", end="   ")
        print()
        
        for i in range(self.n_sardines):
            print(f"S{i+1:<7}", end="")
            for val in self.sardine_random_values[i]:
                print(f"{val:5.3f}", end=" ")
            print()
    
    def print_sequences_and_solutions(self):
        """Print job sequences for each sailfish and sardine"""
        print("\n" + "="*80)
        if self.current_iteration == 0:
            print("3. JOB SEQUENCES FOR EACH SAILFISH AND SARDINE")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 1: GENERATING NEW SEQUENCES")
        print("="*80)
        
        self.sailfish_sequences = []
        self.sardine_sequences = []
        
        print("SAILFISH Job Sequences:")
        for i in range(self.n_sailfish):
            print(f"\n===== SF{i+1} ====================================================")
            
            random_vals = self.sailfish_random_values[i]
            job_sequence, makespan, gantt_chart = convert_random_to_schedule(random_vals, self.jobs_data)
            self.sailfish_sequences.append(job_sequence)
            
            print(f"Random values: {random_vals}")
            
            # Show transformation process - create ranking of each position
            n_positions = len(random_vals)
            value_index_pairs = [(random_vals[j], j) for j in range(n_positions)]
            sorted_pairs = sorted(value_index_pairs)  # Sort by value
            
            # Create ranking: each original position gets its rank in sorted order
            position_ranks = [0] * n_positions
            for rank, (value, original_pos) in enumerate(sorted_pairs):
                position_ranks[original_pos] = rank + 1  # 1-based ranking
            
            print(f"Sorted indices: {position_ranks}")
            # For transformation, use 0-based ranks for modulo operation
            transformed = [(rank - 1) % self.n_jobs + 1 for rank in position_ranks]
            print(f"Transformed ((x mod {self.n_jobs}) + 1): {transformed}")
            print(f"Job sequence: {job_sequence}")
        
        print("\nSARDINE Job Sequences:")
        for i in range(self.n_sardines):
            print(f"\n===== S{i+1} ====================================================")
            
            random_vals = self.sardine_random_values[i]
            job_sequence, makespan, gantt_chart = convert_random_to_schedule(random_vals, self.jobs_data)
            self.sardine_sequences.append(job_sequence)
            
            print(f"Random values: {random_vals}")
            
            # Show transformation process
            value_index_pairs = [(random_vals[j], j + 1) for j in range(len(random_vals))]  # Store 1-based positions
            sorted_pairs = sorted(value_index_pairs)  # Sort by value
            # The sorted indices show the 1-based position of each value in ascending order
            sorted_indices = [pos for val, pos in sorted_pairs]
            print(f"Sorted indices: {sorted_indices}")
            # For transformation, use 0-based indices for modulo operation
            transformed = [((pos - 1) % self.n_jobs) + 1 for val, pos in sorted_pairs]
            print(f"Transformed ((x mod {self.n_jobs}) + 1): {transformed}")
            print(f"Job sequence: {job_sequence}")
    
    def calculate_detailed_fitness(self):
        """Calculate fitness (makespan) for each individual with detailed calculations and tables"""
        print("\n" + "="*80)
        if self.current_iteration == 0:
            print("4. DETAILED FITNESS CALCULATION FOR EACH INDIVIDUAL")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 2: DETAILED FITNESS CALCULATION")
        print("="*80)
        
        self.sailfish_fitness = []
        self.sardine_fitness = []
        
        print("SAILFISH Fitness Calculations:")
        print("=" * 50)
        
        for i, job_sequence in enumerate(self.sailfish_sequences):
            print(f"\nCALCULATING FITNESS FOR SAILFISH SF{i+1}")
            print("-" * 40)
            
            # Create job operations table
            operations_table = create_job_operations_table(job_sequence, self.jobs_data, f"SF{i+1}")
            print("Job Operations Table:")
            print(operations_table.to_string())
            
            # Calculate makespan
            makespan, gantt_chart = calculate_makespan(job_sequence, self.jobs_data, show_details=False)
            self.sailfish_fitness.append(makespan)
            
            # Create Gantt chart DataFrame
            if gantt_chart:
                gantt_df = pd.DataFrame(gantt_chart)
                print(f"\nGantt Chart Details (Makespan: {makespan}):")
                print(gantt_df[['step', 'job', 'operation', 'machine', 'duration', 'start_time', 'finish_time']].to_string(index=False))
            
            if makespan < self.best_fitness:
                self.best_fitness = makespan
                self.best_sequence = job_sequence
                print(f"\n     *** NEW BEST SOLUTION! Makespan: {makespan} ***")
            
            print("-" * 60)
        
        print("\n" + "=" * 50)
        print("SARDINE Fitness Calculations:")
        print("=" * 50)
        
        for i, job_sequence in enumerate(self.sardine_sequences):
            print(f"\nCALCULATING FITNESS FOR SARDINE S{i+1}")
            print("-" * 40)
            
            # Create job operations table
            operations_table = create_job_operations_table(job_sequence, self.jobs_data, f"S{i+1}")
            print("Job Operations Table:")
            print(operations_table.to_string())
            
            # Calculate makespan
            makespan, gantt_chart = calculate_makespan(job_sequence, self.jobs_data, show_details=False)
            self.sardine_fitness.append(makespan)
            
            # Create Gantt chart DataFrame
            if gantt_chart:
                gantt_df = pd.DataFrame(gantt_chart)
                print(f"\nGantt Chart Details (Makespan: {makespan}):")
                print(gantt_df[['step', 'job', 'operation', 'machine', 'duration', 'start_time', 'finish_time']].to_string(index=False))
            
            if makespan < self.best_fitness:
                self.best_fitness = makespan
                self.best_sequence = job_sequence
                print(f"\n     *** NEW BEST SOLUTION! Makespan: {makespan} ***")
            
            print("-" * 60)
    
    def print_fitness_summary(self):
        """Print summary of all fitness scores"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("FITNESS SUMMARY")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 3: FITNESS SUMMARY")
        print("="*80)
        
        print("SAILFISH FITNESS SCORES:")
        print("-" * 30)
        for i, fitness in enumerate(self.sailfish_fitness):
            marker = " ⭐ BEST" if fitness == min(self.sailfish_fitness) else ""
            print(f"SF{i+1}: {fitness}{marker}")
        
        print("\nSARDINE FITNESS SCORES:")
        print("-" * 25)
        for i, fitness in enumerate(self.sardine_fitness):
            marker = " ⭐ BEST" if fitness == min(self.sardine_fitness) else ""
            print(f"S{i+1}: {fitness}{marker}")
        
        print(f"\nOVERALL SUMMARY:")
        print("-" * 20)
        print(f"Best Sailfish Makespan: {min(self.sailfish_fitness)}")
        print(f"Best Sardine Makespan: {min(self.sardine_fitness)}")
        print(f"Overall Best Makespan: {self.best_fitness}")
        print(f"Best Job Sequence: {self.best_sequence}")
        
        # Update elite positions
        best_sailfish_idx = self.sailfish_fitness.index(min(self.sailfish_fitness))
        self.elite_sailfish_fitness = min(self.sailfish_fitness)
        self.elite_sailfish_position = self.sailfish_random_values[best_sailfish_idx].copy()
        
        if self.sardine_fitness:
            best_sardine_idx = self.sardine_fitness.index(min(self.sardine_fitness))
            self.injured_sardine_fitness = min(self.sardine_fitness)
            self.injured_sardine_position = self.sardine_random_values[best_sardine_idx].copy()
    
    def perform_sailfish_sardine_replacement(self):
        """Replace sailfish with better sardines"""
        print(f"\n" + "="*80)
        print(f"ITERATION {self.current_iteration} - STEP 4: SAILFISH-SARDINE REPLACEMENT")
        print("="*80)
        
        worst_sailfish_fitness = max(self.sailfish_fitness)
        better_sardines = []
        
        for i, sardine_fitness in enumerate(self.sardine_fitness):
            if sardine_fitness < worst_sailfish_fitness:
                better_sardines.append((i, sardine_fitness))
        
        print(f"Analysis:")
        print(f"- Worst sailfish makespan: {worst_sailfish_fitness}")
        print(f"- Sardines better than worst sailfish: {len(better_sardines)}")
        
        if not better_sardines:
            print("- No replacement will occur")
            return
        
        better_sardines.sort(key=lambda x: x[1])
        sardines_to_remove = []
        replacements_made = []
        
        for sardine_idx, sardine_fitness in better_sardines:
            worst_sf_idx = self.sailfish_fitness.index(max(self.sailfish_fitness))
            worst_sf_fitness = self.sailfish_fitness[worst_sf_idx]
            
            if sardine_fitness < worst_sf_fitness:
                print(f"\nReplacement {len(replacements_made) + 1}:")
                print(f"- Sardine S{sardine_idx+1} (makespan: {sardine_fitness}) -> Sailfish SF{worst_sf_idx+1} (makespan: {worst_sf_fitness})")
                
                # Replace sailfish with sardine data
                self.sailfish_random_values[worst_sf_idx] = self.sardine_random_values[sardine_idx].copy()
                self.sailfish_sequences[worst_sf_idx] = self.sardine_sequences[sardine_idx]
                self.sailfish_fitness[worst_sf_idx] = self.sardine_fitness[sardine_idx]
                
                sardines_to_remove.append(sardine_idx)
                replacements_made.append({
                    'sardine_idx': sardine_idx,
                    'sailfish_idx': worst_sf_idx,
                    'new_fitness': sardine_fitness,
                    'old_fitness': worst_sf_fitness
                })
                
                if sardine_fitness < self.best_fitness:
                    self.best_fitness = sardine_fitness
                    self.best_sequence = self.sardine_sequences[sardine_idx]
                    print(f"  NEW OVERALL BEST SOLUTION! Makespan: {sardine_fitness}")
            else:
                break
        
        # Remove replaced sardines
        sardines_to_remove.sort(reverse=True)
        for sardine_idx in sardines_to_remove:
            del self.sardine_random_values[sardine_idx]
            del self.sardine_sequences[sardine_idx]
            del self.sardine_fitness[sardine_idx]
            del self.original_sardine_positions[sardine_idx]
            self.n_sardines -= 1
        
        print(f"\nReplacement Summary: {len(replacements_made)} replacements made")
        
        # Update elite positions
        if self.sailfish_fitness:
            best_sailfish_idx = self.sailfish_fitness.index(min(self.sailfish_fitness))
            self.elite_sailfish_fitness = min(self.sailfish_fitness)
            self.elite_sailfish_position = self.sailfish_random_values[best_sailfish_idx].copy()
        
        if self.sardine_fitness:
            best_sardine_idx = self.sardine_fitness.index(min(self.sardine_fitness))
            self.injured_sardine_fitness = min(self.sardine_fitness)
            self.injured_sardine_position = self.sardine_random_values[best_sardine_idx].copy()
        else:
            self.injured_sardine_fitness = self.elite_sailfish_fitness
            self.injured_sardine_position = self.elite_sailfish_position.copy()
    
    def calculate_pd_and_lambda_values(self):
        """Calculate PD and lambda values"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("5. CALCULATE PD AND LAMBDA VALUES")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 5: CALCULATE PD AND LAMBDA VALUES")
        print("="*80)
        
        total_population = self.n_sailfish + self.n_sardines
        self.PD = 1 - (self.n_sailfish / total_population)
        
        print(f"Population Decline (PD) = 1 - ({self.n_sailfish} / {total_population}) = {self.PD:.6f}")
        print()
        
        self.lambda_k_values = []
        print("Lambda Calculations:")
        
        for k in range(self.n_sailfish):
            random_val = round(random.random(), 3)
            lambda_k = (2 * random_val * self.PD) - self.PD
            self.lambda_k_values.append(lambda_k)
            
            print(f"SF{k+1}: λ_{k+1} = (2 × {random_val} × {self.PD:.6f}) - {self.PD:.6f} = {lambda_k:.6f}")
    
    def update_sailfish_positions(self):
        """Update sailfish positions using fitness values"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("6. UPDATE SAILFISH POSITIONS")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 6: UPDATE SAILFISH POSITIONS")
        print("="*80)
        
        print("FORMULA: SF[i] = elite_sailfish_fitness - lambda[k] × ((random × (elite_sailfish_fitness + injured_sardine_fitness)/2) - old_sailfish)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness}")
        print(f"Injured Sardine Fitness: {self.injured_sardine_fitness}")
        print("Using original positions for updates...")
        new_sailfish_positions = []
        
        for k in range(self.n_sailfish):
            print(f"\nUpdating SF{k+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                injured_sardine_fitness = self.injured_sardine_fitness
                old_sailfish_j = self.original_sailfish_positions[k][j]
                
                avg_elite_injured = (elite_sf_fitness + injured_sardine_fitness) / 2
                bracket_term = (rand * avg_elite_injured) - old_sailfish_j
                lambda_term = self.lambda_k_values[k] * bracket_term
                new_val = elite_sf_fitness - lambda_term
                
                # Normalize to [0,1] range
                new_val = max(0, min(1, abs(new_val) % 1))
                new_val = round(new_val, 3)
                new_position.append(new_val)
                
                print(f"  Pos[{j+1}]: {elite_sf_fitness} - {self.lambda_k_values[k]:.6f} × (({rand:.3f} × {avg_elite_injured:.1f}) - {old_sailfish_j:.3f}) = {new_val:.3f}")
            
            new_sailfish_positions.append(new_position)
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        self.sailfish_random_values = new_sailfish_positions
        print("\nAll sailfish positions updated!")
    
    def calculate_ap_and_update_sardines(self):
        """Calculate AP and update sardine positions"""
        print(f"\n" + "="*80)
        if self.current_iteration == 0:
            print("7. CALCULATE AP AND UPDATE SARDINE POSITIONS")
        else:
            print(f"ITERATION {self.current_iteration} - STEP 7: CALCULATE AP AND UPDATE SARDINE POSITIONS")
        print("="*80)
        
        if self.n_sardines == 0:
            print("No sardines remaining. Skipping sardine update.")
            return
        
        self.AP = self.A * (1 - (2 * (self.current_iteration + 1) * self.epsilon))
        
        print(f"Attack Power (AP) = {self.A} × (1 - (2 × {self.current_iteration + 1} × {self.epsilon})) = {self.AP:.6f}")
        
        if self.AP >= 0.5:
            print(f"AP >= 0.5: Update ALL sardine positions")
            self.update_all_sardines()
        else:
            print(f"AP < 0.5: Partial sardine update")
            self.update_partial_sardines()
    
    def update_all_sardines(self):
        """Update all sardine positions when AP >= 0.5 using fitness values"""
        print("\nFORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness}")
        print("Updating ALL sardines:")
        new_sardine_positions = []
        
        for i in range(self.n_sardines):
            print(f"\nUpdating S{i+1}:")
            new_position = []
            
            for j in range(self.problem_size):
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                old_sardine_j = self.original_sardine_positions[i][j]
                
                bracket_term = elite_sf_fitness - old_sardine_j + self.AP
                new_val = rand * bracket_term
                
                # Normalize to [0,1] range
                new_val = max(0, min(1, abs(new_val) % 1))
                new_val = round(new_val, 3)
                new_position.append(new_val)
                
                print(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness} - {old_sardine_j:.3f} + {self.AP:.6f}) = {new_val:.3f}")
            
            new_sardine_positions.append(new_position)
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
        
        self.sardine_random_values = new_sardine_positions
        print("\nAll sardine positions updated!")
    
    def update_partial_sardines(self):
        """Update partial sardines when AP < 0.5 using fitness values"""
        print("\nFORMULA: S[i] = random × (elite_sailfish_fitness - old_sardine + AP)")
        print(f"Elite Sailfish Fitness: {self.elite_sailfish_fitness}")
        print("Partial sardine update:")
        
        alpha = int(self.n_sardines * self.AP)
        beta = int(self.problem_size * self.AP)
        
        print(f"alpha = {self.n_sardines} × {self.AP:.6f} = {alpha}")
        print(f"beta = {self.problem_size} × {self.AP:.6f} = {beta}")
        
        if alpha == 0 or beta == 0:
            print("Alpha or beta is 0, no sardines updated.")
            return
        
        sardines_to_update = random.sample(range(self.n_sardines), min(alpha, self.n_sardines))
        
        print(f"Selected sardines to update: {[f'S{i+1}' for i in sardines_to_update]}")
        
        for i in sardines_to_update:
            print(f"\nUpdating S{i+1} (partial):")
            positions_to_update = random.sample(range(self.problem_size), min(beta, self.problem_size))
            print(f"Updating positions: {[j+1 for j in positions_to_update]}")
            
            new_position = self.original_sardine_positions[i].copy()
            
            for j in positions_to_update:
                rand = round(random.random(), 3)
                elite_sf_fitness = self.elite_sailfish_fitness
                old_sardine_j = self.original_sardine_positions[i][j]
                
                bracket_term = elite_sf_fitness - old_sardine_j + self.AP
                new_val = rand * bracket_term
                
                # Normalize to [0,1] range
                new_val = max(0, min(1, abs(new_val) % 1))
                new_val = round(new_val, 3)
                new_position[j] = new_val
                
                print(f"  Pos[{j+1}]: {rand:.3f} × ({elite_sf_fitness} - {old_sardine_j:.3f} + {self.AP:.6f}) = {new_val:.3f}")
            
            self.sardine_random_values[i] = new_position
            print(f"New position: {[f'{x:.3f}' for x in new_position]}")
    
    def print_comprehensive_results_table(self):
        """Print comprehensive results table"""
        print(f"\n" + "="*120)
        if self.current_iteration == 0:
            print("COMPREHENSIVE RESULTS TABLE")
        else:
            print(f"ITERATION {self.current_iteration} - COMPREHENSIVE RESULTS TABLE")
        print("="*120)
        
        print(f"{'ID':<4} {'Random Values':<35} {'Job Sequence':<25} {'Makespan':<10}")
        print("-" * 120)
        
        # Sailfish table
        for i in range(self.n_sailfish):
            random_str = str([f"{x:.3f}" for x in self.sailfish_random_values[i]][:8]) + "..."  # Show first 8 values
            sequence_str = str(self.sailfish_sequences[i][:15]) + ("..." if len(self.sailfish_sequences[i]) > 15 else "")
            makespan = self.sailfish_fitness[i]
            
            marker = " 🎯" if abs(makespan - self.best_fitness) < 1e-6 else ""
            print(f"SF{i+1:<2} {random_str:<35} {sequence_str:<25} {makespan:<10}{marker}")
        
        # Sardine table
        for i in range(self.n_sardines):
            random_str = str([f"{x:.3f}" for x in self.sardine_random_values[i]][:8]) + "..."  # Show first 8 values
            sequence_str = str(self.sardine_sequences[i][:15]) + ("..." if len(self.sardine_sequences[i]) > 15 else "")
            makespan = self.sardine_fitness[i]
            
            marker = " 🎯" if abs(makespan - self.best_fitness) < 1e-6 else ""
            print(f"S{i+1:<3} {random_str:<35} {sequence_str:<25} {makespan:<10}{marker}")
        
        print(f"\nBest Solution: {self.best_sequence} with makespan: {self.best_fitness}")
    
    def run_iteration_zero(self):
        """Run the initial iteration"""
        self.current_iteration = 0
        
        self.print_initial_parameters()
        self.print_random_populations()
        self.save_original_positions()
        self.print_sequences_and_solutions()
        self.calculate_detailed_fitness()
        self.print_fitness_summary()
        self.print_comprehensive_results_table()
        self.calculate_pd_and_lambda_values()
        self.update_sailfish_positions()
        self.calculate_ap_and_update_sardines()
        
        self.fitness_history.append(self.best_fitness)
        
        print(f"\n" + "="*80)
        print("ITERATION 0 COMPLETED")
        print("="*80)
        print(f"Best makespan: {self.best_fitness}")
        print(f"Best sequence: {self.best_sequence}")
    
    def run_iteration(self, iteration_num):
        """Run a single iteration"""
        self.current_iteration = iteration_num
        
        print(f"\n" + "="*100)
        print(f"STARTING ITERATION {iteration_num}")
        print("="*100)
        
        self.save_original_positions()
        self.print_sequences_and_solutions()
        self.calculate_detailed_fitness()
        self.print_fitness_summary()
        self.perform_sailfish_sardine_replacement()
        self.print_comprehensive_results_table()
        self.calculate_pd_and_lambda_values()
        self.update_sailfish_positions()
        self.calculate_ap_and_update_sardines()
        
        self.fitness_history.append(self.best_fitness)
        
        print(f"\n" + "="*80)
        print(f"ITERATION {iteration_num} COMPLETED")
        print("="*80)
        print(f"Best makespan: {self.best_fitness}")
        print(f"Best sequence: {self.best_sequence}")
    
    def check_convergence(self):
        """Check if algorithm has converged"""
        if len(self.fitness_history) < 2:
            return False
        
        improvement = abs(self.fitness_history[-2] - self.fitness_history[-1])
        return improvement < self.epsilon
    
    def run_optimization(self):
        """Run the complete optimization process"""
        print("STARTING SAILFISH JOB SHOP SCHEDULING OPTIMIZATION ALGORITHM")
        print("="*80)
        
        self.run_iteration_zero()
        
        for iteration in range(1, self.max_iter + 1):
            if self.n_sardines == 0:
                print(f"\nNo sardines remaining after iteration {iteration-1}. Stopping.")
                break
            
            self.run_iteration(iteration)
        
        self.print_final_results()
    
    def print_final_results(self):
        """Print final optimization results"""
        print(f"\n" + "="*100)
        print("FINAL JOB SHOP SCHEDULING OPTIMIZATION RESULTS")
        print("="*100)
        
        print(f"Algorithm Parameters:")
        print(f"- Initial Sailfish: {self.original_n_sailfish}")
        print(f"- Initial Sardines: {self.original_n_sardines}")
        print(f"- Final Sailfish: {self.n_sailfish}")
        print(f"- Final Sardines: {self.n_sardines}")
        print(f"- Iterations: {len(self.fitness_history)}")
        print()
        
        print(f"Job Shop Parameters:")
        print(f"- Number of Jobs: {self.n_jobs}")
        print(f"- Total Operations: {self.total_operations}")
        print(f"- Problem Size: {self.problem_size}")
        print()
        
        print(f"Best Solution:")
        print(f"- Job Sequence: {self.best_sequence}")
        print(f"- Makespan: {self.best_fitness}")
        print()
        
        if self.best_sequence:
            print("Detailed Best Solution Analysis:")
            makespan, gantt_chart = calculate_makespan(self.best_sequence, self.jobs_data, show_details=True)
            
            # Create final operations table for best solution
            best_operations_table = create_job_operations_table(self.best_sequence, self.jobs_data, "Best Solution")
            print("\nBest Solution Operations Table:")
            print(best_operations_table.to_string())
            
            print("\nGantt Chart for Best Solution:")
            if gantt_chart:
                gantt_df = pd.DataFrame(gantt_chart)
                print(gantt_df[['step', 'job', 'operation', 'machine', 'duration', 'start_time', 'finish_time']].to_string(index=False))
        
        print(f"\nMakespan Evolution:")
        print(f"- Initial: {self.fitness_history[0]}")
        print(f"- Final: {self.fitness_history[-1]}")
        print(f"- Improvement: {self.fitness_history[0] - self.fitness_history[-1]}")
        print(f"- History: {self.fitness_history}")
        
        print("\n" + "="*100)
        print("JOB SHOP SCHEDULING OPTIMIZATION COMPLETED!")
        print("="*100)


def create_sample_jsp_csv():
    """Create sample Job Shop CSV file for testing"""
    jobs_data = [
        ['Job', 'Operasi 1 Mesin', 'Operasi 1 Durasi', 'Operasi 2 Mesin', 'Operasi 2 Durasi', 'Operasi 3 Mesin', 'Operasi 3 Durasi'],
        [1, 1, 3, 2, 3, 3, 2],
        [2, 1, 1, 3, 5, 2, 3],
        [3, 2, 3, 1, 2, 3, 3]
    ]
    
    with open('jobs.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(jobs_data)
    
    print("Sample Job Shop CSV file created: jobs.csv")


def main():
    """Interactive CLI to run the Sailfish Job Shop Scheduling Optimizer"""
    # Constants
    A_CONST = 4
    EPSILON_CONST = 0.001

    # Defaults
    DEFAULT_PARAMS = {
        "n_sailfish": 3,
        "n_sardines": 5,
        "max_iter": 10,
    }

    # 1) Dataset selection
    dataset_path = ask_dataset_selection()

    # 2) Logging selection
    log_to_file = ask_logging_selection()

    # 3) Parameter selection (A and epsilon are constants)
    params = ask_parameter_selection(DEFAULT_PARAMS, A_CONST, EPSILON_CONST)

    # Load data
    if os.path.isfile(dataset_path):
        jobs_data = read_jsp_data_from_csv(dataset_path)
    else:
        jobs_data = read_jsp_data_from_csv("jobs.csv")

    # Run with chosen logging mode
    if log_to_file:
        # Let optimizer handle file-only logging via OutputLogger
        optimizer = SailfishJSPOptimizer(
            n_sailfish=params["n_sailfish"],
            n_sardines=params["n_sardines"],
            jobs_data=jobs_data,
            max_iter=params["max_iter"],
            A=params["A"],
            epsilon=params["epsilon"],
            log_to_file=True,
        )
        optimizer.run_optimization()
        # No terminal output; full details are in the generated log file.
    else:
        # Silence all intermediate prints and show only conclusion
        optimizer = SailfishJSPOptimizer(
            n_sailfish=params["n_sailfish"],
            n_sardines=params["n_sardines"],
            jobs_data=jobs_data,
            max_iter=params["max_iter"],
            A=params["A"],
            epsilon=params["epsilon"],
            log_to_file=False,
        )
        with StdoutSilencer():
            optimizer.run_optimization()
        # Print concise conclusion
        print("Conclusion:")
        print(f"- Best makespan: {optimizer.best_fitness}")
        print(f"- Best sequence: {optimizer.best_sequence}")
        print(f"- Iterations run: {len(optimizer.fitness_history)}")
        # Explain the best solution and sequence mapping
        if optimizer.best_sequence:
            print("- Explanation: best sequence lists jobs in the order operations are scheduled.")
            print("  Repeating job IDs indicate successive operations of the same job." )
            # Provide first few scheduled steps as illustration
            ms, gantt = calculate_makespan(optimizer.best_sequence, optimizer.jobs_data, show_details=False)
            if gantt:
                preview = gantt[:min(10, len(gantt))]
                print("  First steps (job, op -> machine [start-finish]):")
                for e in preview:
                    print(f"  - J{e['job']} Op{e['operation']} -> M{e['machine']} [{e['start_time']}-{e['finish_time']}]")


if __name__ == "__main__":
    main()