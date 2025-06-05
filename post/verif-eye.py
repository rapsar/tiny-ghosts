import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button



def main(folder):
    # Ensure rejected subfolder exists
    rejected_folder = os.path.join(folder, "_rejected")
    os.makedirs(rejected_folder, exist_ok=True)

    # At startup, optionally move all images into rejected folder
    if REJECTED_BY_DEFAULT:
        for name in os.listdir(folder):
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                shutil.move(os.path.join(folder, name), os.path.join(rejected_folder, name))

    # Collect image filenames from main folder and rejected subfolder
    main_images = [f for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    rejected_images = [f for f in os.listdir(rejected_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    # Combine and sort all filenames, tracking their status
    all_names = sorted(set(main_images + rejected_images))
    images = [(name, 'rejected' if name in rejected_images else 'main') for name in all_names]

    if not images:
        print(f"No images found in folder: {folder}")
        return

    idx = 0
    use_jet = False
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0, right=1, bottom=0.1)

    def show_image():
        ax.clear()
        name, status = images[idx]
        filepath = os.path.join(folder if status == 'main' else rejected_folder, name)
        img = mpimg.imread(filepath)
        # Crop to top 1280 rows
        img = img[:1280, :, :]

        # Use red channel as grayscale data
        data = img[:, :, 0]
        ax.imshow(data, cmap='jet' if use_jet else 'gray')
        ax.set_title(f"{name} ({idx+1}/{len(images)})")
        ax.axis('off')

        # Highlight Accept/Reject buttons based on status
        if status == 'rejected':
            reject_btn.ax.set_facecolor('red')
            accept_btn.ax.set_facecolor(default_accept_color)
        else:
            accept_btn.ax.set_facecolor('green')
            reject_btn.ax.set_facecolor(default_reject_color)

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx, use_jet
        if event.key == 'right':
            idx = (idx + 1) % len(images)
            show_image()
        elif event.key == 'left':
            idx = (idx - 1) % len(images)
            show_image()
        elif event.key == 'up':
            on_accept(event)
        elif event.key == 'down':
            on_reject(event)
        elif event.key == 'shift':
            use_jet = not use_jet
            show_image()
        elif event.key == 'escape':
            plt.close(fig)

    # Connect keypress events
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Add Accept and Reject buttons
    accept_ax = fig.add_axes([0.3, 0.01, 0.1, 0.05])
    reject_ax = fig.add_axes([0.6, 0.01, 0.1, 0.05])
    accept_btn = Button(accept_ax, 'Accept')
    reject_btn = Button(reject_ax, 'Reject')
    # Store default button colors for later
    default_accept_color = accept_btn.ax.get_facecolor()
    default_reject_color = reject_btn.ax.get_facecolor()

    def on_accept(event):
        nonlocal idx
        name, status = images[idx]
        if status == 'rejected':
            shutil.move(os.path.join(rejected_folder, name), os.path.join(folder, name))
            images[idx] = (name, 'main')
        show_image()

    def on_reject(event):
        nonlocal idx
        name, status = images[idx]
        if status == 'main':
            shutil.move(os.path.join(folder, name), os.path.join(rejected_folder, name))
            images[idx] = (name, 'rejected')
        show_image()

    accept_btn.on_clicked(on_accept)
    reject_btn.on_clicked(on_reject)

    # Initial display
    show_image()
    plt.show()


if __name__ == '__main__':
    # Specify your folder path here:
    folder_path = "/Users/rss367/Desktop/2024bww/Muleshoe/results/UpperWildcat/dusk/_41mini/flash"
    # Set default rejection behavior for this run
    REJECTED_BY_DEFAULT = False  # Change to True to reject images by default
    main(folder_path)
