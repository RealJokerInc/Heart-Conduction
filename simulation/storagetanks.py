# -*- coding: utf-8 -*-
"""Storage-tank grid simulation (John Zimmerman, 2026-04-24).

2D grid of tanks connected to 8 neighbours (Moore neighbourhood). Each step,
tanks above a pumping threshold push fluid toward lower-volume neighbours.
Inlets are held at max volume, outlets at zero.

Boundary-effect analog: interior tanks have 8 neighbours; edge/corner tanks
have 5/3. Fewer sinks per source at the boundary means the edge fills/empties
at a different rate than the bulk — the storage-tank counterpart of reduced
electrotonic loading at a bath-coupled cardiac boundary (Kleber speedup).

Original: https://colab.research.google.com/drive/19X48Z8hPbodYucLOVkGqPLJ5Ewc1oue3
"""

from pathlib import Path

import cv2
import numpy as np


class StorageTank:
    def __init__(self, id, max_volume, threshold=45.0, maxpump=10.0):
        self.id = id
        self.max_volume = max_volume
        self.current_volume = 0
        self.virtual_volume = 0
        self.connections = list()
        self.threshold = threshold
        self.max_pump = maxpump
        self.pumpfactor = np.sqrt((self.max_volume - self.threshold))
        self.isInlet = False
        self.isOutlet = False

    def checkpump(self, tank):
        if self.current_volume > self.threshold:
            if self.current_volume > tank.current_volume:
                pump_amount = StorageTank.GetPumpAmount(
                    self.current_volume, self.threshold, self.pumpfactor, self.max_pump
                )
                if pump_amount > np.abs(self.current_volume - tank.current_volume):
                    midpoint = np.average([self.current_volume, tank.current_volume])
                    pump_amount = (midpoint - tank.current_volume) / 2
                self.virtual_volume -= pump_amount
                tank.virtual_volume += pump_amount

    @staticmethod
    def GetPumpAmount(currentVolume, threshold, pumpfactor, max_pump):
        amount = (np.sqrt((currentVolume - threshold)) / pumpfactor) * max_pump
        if amount > max_pump:
            amount = max_pump
        elif amount < 0.0:
            amount = 0.0
        return amount

    def add_connection(self, connection):
        if connection not in self.connections:
            self.connections.append(connection)

    def remove_connection(self, connection):
        if connection in self.connections:
            self.connections.remove(connection)


def GenerateStorageGrid(gridx_dim, gridy_dim, maxvolume, threshold, maxpump, inletIds, outletIds):
    tankList = []
    tank_grid_2d = [[None for _ in range(gridy_dim)] for _ in range(gridx_dim)]

    tank_id_counter = 0
    for x in range(gridx_dim):
        for y in range(gridy_dim):
            tank = StorageTank(tank_id_counter, maxvolume, threshold)
            tank_grid_2d[x][y] = tank
            tankList.append(tank)
            if tank_id_counter in inletIds:
                tank.isInlet = True
            if tank_id_counter in outletIds:
                tank.isOutlet = True
            tank_id_counter += 1

    for x in range(gridx_dim):
        for y in range(gridy_dim):
            current_tank = tank_grid_2d[x][y]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gridx_dim and 0 <= ny < gridy_dim:
                        current_tank.add_connection(tank_grid_2d[nx][ny])

    return tankList


def RunSimulation(tankList, gridx, gridy, steps=1000, output_path="outputs/tank_simulation.mp4"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    scale_factor = 5
    display_width = gridx * scale_factor
    display_height = gridy * scale_factor
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (display_width, display_height))

    for step in range(steps):
        if step % 50 == 0:
            print(f"Step {step}/{steps}")

        for tank in tankList:
            tank.virtual_volume = 0

        for tank in tankList:
            for connection in tank.connections:
                connection.checkpump(tank)

        for tank in tankList:
            if tank.isInlet:
                tank.current_volume = tank.max_volume
            elif tank.isOutlet:
                tank.current_volume = 0.0
            else:
                tank.current_volume += tank.virtual_volume
                if tank.current_volume > tank.max_volume:
                    tank.current_volume = tank.max_volume
            tank.virtual_volume = 0.0

        ZZ = np.zeros((gridy, gridx), dtype=np.float32)
        for tank in tankList:
            col_idx = tank.id // gridy
            row_idx = tank.id % gridy
            ZZ[row_idx, col_idx] = tank.current_volume

        ZZ_normalized = (ZZ / 100.0 * 255).astype(np.uint8)
        img_colored = cv2.applyColorMap(ZZ_normalized, cv2.COLORMAP_JET)
        img_resized = cv2.resize(
            img_colored, (display_width, display_height), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            img_resized, f"Step: {step}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
        )
        video_writer.write(img_resized)

    video_writer.release()
    print(f"Simulation video saved as {output_path}")


if __name__ == "__main__":
    gridx = 80
    gridy = 50
    steps = 2000
    threshold = 45.0
    maxvolume = 100.0
    maxpump = 5.0

    # Inlets / outlets (PI's original configuration — sparse inlet cluster)
    inletIds = np.array([703, 705, 706, 707])
    outletIds = np.arange(gridx * gridy - gridy, gridx * gridy)

    tankList = GenerateStorageGrid(
        gridx, gridy, maxvolume, threshold, maxpump, inletIds, outletIds
    )

    for tank in tankList[1].connections:
        print(
            f"Tank ID: {tank.id}, Current Volume: {tank.current_volume},"
            f" Max Volume: {tank.max_volume}"
        )

    RunSimulation(tankList, gridx, gridy, steps=steps)
