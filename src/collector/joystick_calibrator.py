"""
Module pour la calibration du joystick.

Ce module fournit une interface graphique pour calibrer le joystick
et enregistrer les valeurs maximales/minimales pour chaque axe.
"""

import pygame
import time
from utils_collector import joystick_calibration


class JoystickCalibratorUI:
    """Interface graphique pour calibrer le joystick."""

    def __init__(self, joystick):
        """
        Initialise l'interface de calibration.

        Args:
            joystick: L'instance de joystick Pygame à calibrer
        """
        self.joystick = joystick
        if not joystick:
            print("[ERREUR] Aucun joystick détecté pour la calibration")
            return

        # Initialisation de Pygame
        if not pygame.get_init():
            pygame.init()

        # Création de la fenêtre
        self.width, self.height = 800, 600
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Calibration du Joystick")

        # Couleurs
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # Polices
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 24)

        # Variables pour la calibration
        self.steering_min = 0.0
        self.steering_max = 0.0
        self.accel_min = 0.0
        self.accel_max = 0.0

        # État de la calibration
        self.is_calibrating = True
        self.start_time = time.time()
        self.calibration_duration = 10  # Durée en secondes

    def update_calibration_values(self):
        """Met à jour les valeurs min/max selon les entrées actuelles."""
        if not self.joystick:
            return

        # Lecture des valeurs actuelles
        steering_value = self.joystick.get_axis(0)
        accel_value = self.joystick.get_axis(1)

        # Mise à jour des valeurs min/max pour la direction
        if steering_value < self.steering_min:
            self.steering_min = steering_value
        if steering_value > self.steering_max:
            self.steering_max = steering_value

        # Mise à jour des valeurs min/max pour l'accélération
        if accel_value < self.accel_min:
            self.accel_min = accel_value
        if accel_value > self.accel_max:
            self.accel_max = accel_value

    def save_calibration(self):
        """Enregistre les valeurs de calibration."""
        # Vérification des valeurs
        if self.steering_min == self.steering_max:
            self.steering_min = -1.0
            self.steering_max = 1.0

        if self.accel_min == self.accel_max:
            self.accel_min = -1.0
            self.accel_max = 1.0

        # Mise à jour et sauvegarde
        joystick_calibration.calib_data["steering"]["min"] = self.steering_min
        joystick_calibration.calib_data["steering"]["max"] = self.steering_max
        joystick_calibration.calib_data["acceleration"]["min"] = self.accel_min
        joystick_calibration.calib_data["acceleration"]["max"] = self.accel_max

        joystick_calibration.save_calibration()

    def draw_axis_visualization(self, x, y, width, height, value, min_val, max_val, label):
        """Dessine la visualisation d'un axe."""
        # Cadre
        pygame.draw.rect(self.window, self.WHITE, (x, y, width, height), 2)

        # Position actuelle
        normalized = 0.5
        if min_val != max_val:
            if value >= 0:
                normalized = 0.5 + (value / max_val) * 0.5 if max_val != 0 else 0.5
            else:
                normalized = 0.5 - (value / min_val) * 0.5 if min_val != 0 else 0.5

        pos_x = x + int(width * normalized)
        pygame.draw.line(self.window, self.RED, (pos_x, y), (pos_x, y + height), 3)

        # Texte
        text = self.small_font.render(
            f"{label}: {value:.2f} (Min: {min_val:.2f}, Max: {max_val:.2f})",
            True, self.WHITE
        )
        self.window.blit(text, (x, y - 25))

        # Marques pour -1, 0, 1
        zero_x = x + width // 2
        pygame.draw.line(self.window, self.GREEN, (zero_x, y), (zero_x, y + height), 1)

        mark_min = x + width // 4
        mark_max = x + 3 * (width // 4)
        pygame.draw.line(self.window, self.BLUE, (mark_min, y), (mark_min, y + height), 1)
        pygame.draw.line(self.window, self.BLUE, (mark_max, y), (mark_max, y + height), 1)

    def draw_joystick_position(self, x, y, size):
        """Dessine une représentation visuelle de la position du joystick."""
        if not self.joystick:
            return

        # Cadre
        pygame.draw.rect(
            self.window,
            self.WHITE,
            (x - size//2, y - size//2, size, size),
            2
        )

        # Axes
        pygame.draw.line(
            self.window,
            self.WHITE,
            (x - size//2, y),
            (x + size//2, y),
            1
        )
        pygame.draw.line(
            self.window,
            self.WHITE,
            (x, y - size//2),
            (x, y + size//2),
            1
        )

        # Position actuelle
        steering = self.joystick.get_axis(0)
        accel = -self.joystick.get_axis(1)  # Inversion pour l'intuition

        pos_x = x + int(steering * size//2)
        pos_y = y - int(accel * size//2)

        # Cercle représentant la position
        pygame.draw.circle(self.window, self.RED, (pos_x, pos_y), 10)

    def draw_interface(self, elapsed_time):
        """Dessine l'interface complète de calibration."""
        # Effacer l'écran
        self.window.fill(self.BLACK)

        # Titre
        title = self.font.render("Calibration du Joystick", True, self.WHITE)
        self.window.blit(title, (self.width//2 - title.get_width()//2, 50))

        # Instructions
        instructions = [
            "Déplacez le joystick dans toutes les directions extrêmes",
            f"Temps restant: {max(0, self.calibration_duration - elapsed_time):.1f}s",
            "Appuyez sur ENTRÉE pour terminer ou ÉCHAP pour annuler"
        ]

        for i, text in enumerate(instructions):
            rendered = self.small_font.render(text, True, self.WHITE)
            y_pos = 100 + i * 30
            self.window.blit(rendered, (self.width//2 - rendered.get_width()//2, y_pos))

        # Visualisation des axes
        self.draw_axis_visualization(
            100, 250, self.width - 200, 40,
            self.joystick.get_axis(0),
            self.steering_min, self.steering_max,
            "Direction"
        )

        self.draw_axis_visualization(
            100, 350, self.width - 200, 40,
            self.joystick.get_axis(1),
            self.accel_min, self.accel_max,
            "Accélération"
        )

        # Visualisation du joystick
        self.draw_joystick_position(self.width//2, 480, 200)

    def check_events(self):
        """
        Vérifie les événements utilisateur.

        Returns:
            bool: True si la calibration doit continuer, False sinon
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_RETURN:
                    self.save_calibration()
                    return False
        return True

    def run(self):
        """Exécute la boucle principale de l'interface de calibration."""
        if not self.joystick:
            return False

        clock = pygame.time.Clock()

        while self.is_calibrating:
            # Gestion des événements
            self.is_calibrating = self.check_events()

            # Mise à jour des valeurs de calibration
            self.update_calibration_values()

            # Vérification du temps écoulé
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.calibration_duration:
                self.save_calibration()
                self.is_calibrating = False

            # Dessin de l'interface
            self.draw_interface(elapsed_time)

            # Mise à jour de l'affichage
            pygame.display.flip()
            clock.tick(30)

        # Nettoyage - ne pas fermer la fenêtre avec set_mode mais avec flip et delay
        if pygame.display.get_init():
            pygame.display.flip()
            pygame.time.delay(100)  # Délai pour permettre à pygame de traiter

        return True


def calibrate_joystick(joystick):
    """
    Lance l'interface de calibration pour un joystick.

    Args:
        joystick: Instance du joystick Pygame à calibrer

    Returns:
        bool: True si la calibration a été effectuée, False sinon
    """
    if not joystick:
        print("[ERREUR] Aucun joystick à calibrer")
        return False

    # Sauvegarder l'ID du joystick pour vérification
    try:
        joystick_name = joystick.get_name()
        joystick_id = joystick.get_id()
    except (pygame.error, AttributeError):
        print("[AVERTISSEMENT] Impossible d'obtenir les infos du joystick")
        joystick_name = "Unknown"
        joystick_id = -1

    # Lancer la calibration
    calibrator = JoystickCalibratorUI(joystick)
    result = calibrator.run()

    print("[INFO] Calibration terminée, nettoyage des événements pygame...")

    # S'assurer que pygame est toujours initialisé
    if not pygame.get_init():
        pygame.init()

    # Vider complètement la file d'attente d'événements
    pygame.event.clear()

    # Laisser pygame gérer les événements
    for _ in range(20):
        pygame.event.pump()
        pygame.time.delay(10)

    return result
