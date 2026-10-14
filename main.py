
from datetime import date
from decimal import Decimal
from io import BytesIO
import secrets

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import text
from sqlalchemy.orm import Session

from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from database import get_db, test_db_connection

import logging


from pathlib import Path
import arabic_reshaper
from bidi.algorithm import get_display

from reportlab.platypus import Image as PDFImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Hotel Management API",
    description="Backend API for Al Hayat Hotel Sharjah",
    version="1.1.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8083"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
def startup_event():
    test_db_connection()


# ---------------------------------------------------------
# General endpoints
# ---------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Hotel Management API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------
# Room categories
# ---------------------------------------------------------

@app.get("/api/room-categories")
def get_room_categories(db: Session = Depends(get_db)):
    query = text("""
        SELECT
            id,
            name,
            description,
            price_per_night,
            total_rooms,
            max_guests
        FROM room_categories
        ORDER BY id
    """)

    rows = db.execute(query).mappings().all()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "price_per_night": float(row["price_per_night"]),
            "currency": "AED",
            "total_rooms": row["total_rooms"],
            "max_guests": row["max_guests"],
        }
        for row in rows
    ]


# ---------------------------------------------------------
# Availability
# ---------------------------------------------------------

@app.get("/api/availability")
def check_availability(
    check_in: date = Query(..., description="YYYY-MM-DD"),
    check_out: date = Query(..., description="YYYY-MM-DD"),
    guests: int = Query(1, ge=1, le=4),
    db: Session = Depends(get_db),
):
    if check_in < date.today():
        raise HTTPException(
            status_code=400,
            detail="Check-in cannot be in the past",
        )

    if check_out <= check_in:
        raise HTTPException(
            status_code=400,
            detail="Check-out must be after check-in",
        )

    query = text("""
        SELECT
            rc.id,
            rc.name,
            rc.price_per_night,
            rc.total_rooms,
            rc.max_guests,
            rc.total_rooms - COUNT(b.id) AS available_rooms
        FROM room_categories rc
        LEFT JOIN bookings b
            ON b.room_category_id = rc.id
            AND b.status IN ('pending', 'confirmed')
            AND b.check_in < :check_out
            AND b.check_out > :check_in
        GROUP BY
            rc.id,
            rc.name,
            rc.price_per_night,
            rc.total_rooms,
            rc.max_guests
        HAVING
            rc.max_guests >= :guests
            AND rc.total_rooms - COUNT(b.id) > 0
        ORDER BY rc.price_per_night, rc.id
    """)

    rows = db.execute(
        query,
        {
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
        },
    ).mappings().all()

    return {
        "check_in": check_in.isoformat(),
        "check_out": check_out.isoformat(),
        "guests": guests,
        "available_categories": [
            {
                "id": row["id"],
                "name": row["name"],
                "price_per_night": float(row["price_per_night"]),
                "currency": "AED",
                "available_rooms": row["available_rooms"],
                "max_guests": row["max_guests"],
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------
# Booking request model
# ---------------------------------------------------------

class BookingRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    address: str | None = None

    room_category_id: int
    check_in: date
    check_out: date

    adults: int = Field(ge=1, le=4)
    children: int = Field(ge=0, le=4)
    guests: int = Field(ge=1, le=8)


# ---------------------------------------------------------
# Create booking
# ---------------------------------------------------------

@app.post("/api/bookings", status_code=201)
def create_booking(
    request: BookingRequest,
    db: Session = Depends(get_db),
):
    if request.check_in < date.today():
        raise HTTPException(
            status_code=400,
            detail="Check-in cannot be in the past",
        )

    if request.check_out <= request.check_in:
        raise HTTPException(
            status_code=400,
            detail="Check-out must be after check-in",
        )

    if not request.name.strip() or not request.phone.strip():
        raise HTTPException(
            status_code=400,
            detail="Name and phone cannot be blank",
        )

    total_guests = request.adults + request.children

    if request.guests is not None and request.guests != total_guests:
        raise HTTPException(
            status_code=400,
            detail="Guest count must equal adults plus children",
        )

    nights = (request.check_out - request.check_in).days

    try:
        # Lock the category row to serialize booking attempts
        # for the same room category.
        with db.begin():
            category = db.execute(
                text("""
                    SELECT
                        id,
                        name,
                        price_per_night,
                        total_rooms,
                        max_guests
                    FROM room_categories
                    WHERE id = :category_id
                    FOR UPDATE
                """),
                {"category_id": request.room_category_id},
            ).mappings().first()

            if category is None:
                raise HTTPException(
                    status_code=404,
                    detail="Room category not found",
                )

            # Room-specific adult and child limits
            category_name = category["name"]

            if category_name == "Standard Connecting Family Room":
                max_adults = 4
                max_children = 4
            else:
                max_adults = 2
                max_children = 2

            if request.adults > max_adults:
                raise HTTPException(
                    status_code=400,
                    detail=f"{category_name} allows a maximum of {max_adults} adults",
                )

            if request.children > max_children:
                raise HTTPException(
                    status_code=400,
                    detail=f"{category_name} allows a maximum of {max_children} children",
                )

            if total_guests > category["max_guests"]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{category['name']} allows a maximum of "
                        f"{category['max_guests']} total guests"
                    ),
                )

            booked_count = db.execute(
                text("""
                    SELECT COUNT(*)
                    FROM bookings
                    WHERE room_category_id = :category_id
                      AND status IN ('pending', 'confirmed')
                      AND check_in < :check_out
                      AND check_out > :check_in
                """),
                {
                    "category_id": request.room_category_id,
                    "check_in": request.check_in,
                    "check_out": request.check_out,
                },
            ).scalar_one()

            available_rooms = category["total_rooms"] - booked_count

            if available_rooms <= 0:
                raise HTTPException(
                    status_code=409,
                    detail="No rooms available for these dates",
                )

            # The database room rate is authoritative.
            # Payment and tax processing are intentionally not included.
            total_amount = (
                Decimal(str(category["price_per_night"])) * nights
            )

            customer_id = db.execute(
                text("""
                    INSERT INTO customers
                        (name, email, phone, address)
                    VALUES
                        (:name, :email, :phone, :address)
                    RETURNING id
                """),
                {
                    "name": request.name.strip(),
                    "email": str(request.email),
                    "phone": request.phone.strip(),
                    "address": request.address,
                },
            ).scalar_one()

            booking_reference = "HTL-" + secrets.token_hex(6).upper()

            booking = db.execute(
                text("""
                    INSERT INTO bookings (
                        customer_id,
                        room_category_id,
                        check_in,
                        check_out,
                        status,
                        total_amount,
                        booking_reference,
                        adults,
                        children
                    )
                    VALUES (
                        :customer_id,
                        :category_id,
                        :check_in,
                        :check_out,
                        'pending',
                        :total_amount,
                        :booking_reference,
                        :adults,
                        :children
                    )
                    RETURNING
                        id,
                        booking_reference,
                        status,
                        check_in,
                        check_out,
                        total_amount
                """),
                {
                    "customer_id": customer_id,
                    "category_id": request.room_category_id,
                    "check_in": request.check_in,
                    "check_out": request.check_out,
                    "total_amount": total_amount,
                    "booking_reference": booking_reference,
                    "adults": request.adults,
                    "children": request.children,
                },
            ).mappings().one()

        return {
            "message": "Booking created. Payment is still pending.",
            "booking_id": booking["id"],
            "booking_reference": booking["booking_reference"],
            "status": booking["status"],
            "customer_id": customer_id,
            "room_category": category["name"],
            "check_in": booking["check_in"].isoformat(),
            "check_out": booking["check_out"].isoformat(),
            "nights": nights,
            "adults": request.adults,
            "children": request.children,
            "guests": total_guests,
            "total_amount": float(booking["total_amount"]),
            "currency": "AED",
            "available_rooms_after_booking": available_rooms - 1,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create hotel booking")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Booking could not be created. Check backend logs."
    )


# ---------------------------------------------------------
# Get booking by reference and email
# ---------------------------------------------------------

@app.get("/api/bookings/{booking_reference}")
def get_booking(
    booking_reference: str,
    email: EmailStr,
    db: Session = Depends(get_db),
):
    query = text("""
        SELECT
            b.id,
            b.booking_reference,
            b.status,
            b.check_in,
            b.check_out,
            b.total_amount,
            b.adults,
            b.children,
            c.name AS customer_name,
            c.email AS customer_email,
            rc.name AS room_category,
            rc.price_per_night
        FROM bookings b
        JOIN customers c ON c.id = b.customer_id
        JOIN room_categories rc ON rc.id = b.room_category_id
        WHERE b.booking_reference = :reference
          AND LOWER(c.email) = LOWER(:email)
    """)

    booking = db.execute(
        query,
        {
            "reference": booking_reference,
            "email": str(email),
        },
    ).mappings().first()

    if booking is None:
        raise HTTPException(
            status_code=404,
            detail="Booking not found",
        )

    return {
        "booking_id": booking["id"],
        "booking_reference": booking["booking_reference"],
        "status": booking["status"],
        "customer_name": booking["customer_name"],
        "room_category": booking["room_category"],
        "check_in": booking["check_in"].isoformat(),
        "check_out": booking["check_out"].isoformat(),
        "adults": booking["adults"],
        "children": booking["children"],
        "guests": booking["adults"] + booking["children"],
        "total_amount": float(booking["total_amount"]),
        "currency": "AED",
    }


# ---------------------------------------------------------
# Cancel booking
# ---------------------------------------------------------

class CancellationRequest(BaseModel):
    email: EmailStr


@app.post("/api/bookings/{booking_reference}/cancel")
def cancel_booking(
    booking_reference: str,
    request: CancellationRequest,
    db: Session = Depends(get_db),
):
    try:
        with db.begin():
            category_id = db.execute(
                text("""
                    SELECT b.room_category_id
                    FROM bookings b
                    JOIN customers c ON c.id = b.customer_id
                    WHERE b.booking_reference = :reference
                      AND LOWER(c.email) = LOWER(:email)
                """),
                {
                    "reference": booking_reference,
                    "email": str(request.email),
                },
            ).scalar_one_or_none()

            if category_id is None:
                raise HTTPException(
                    status_code=404,
                    detail="Booking not found",
                )

            # Use the same lock as booking creation.
            db.execute(
                text("""
                    SELECT id
                    FROM room_categories
                    WHERE id = :category_id
                    FOR UPDATE
                """),
                {"category_id": category_id},
            ).first()

            booking = db.execute(
                text("""
                    SELECT b.id, b.status
                    FROM bookings b
                    JOIN customers c ON c.id = b.customer_id
                    WHERE b.booking_reference = :reference
                      AND LOWER(c.email) = LOWER(:email)
                    FOR UPDATE OF b
                """),
                {
                    "reference": booking_reference,
                    "email": str(request.email),
                },
            ).mappings().first()

            if booking is None:
                raise HTTPException(
                    status_code=404,
                    detail="Booking not found",
                )

            if booking["status"] == "cancelled":
                raise HTTPException(
                    status_code=409,
                    detail="Booking is already cancelled",
                )

            if booking["status"] != "pending":
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Only pending bookings can be cancelled "
                        "through this endpoint"
                    ),
                )

            db.execute(
                text("""
                    UPDATE bookings
                    SET status = 'cancelled',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :booking_id
                """),
                {"booking_id": booking["id"]},
            )

        return {
            "message": "Booking cancelled successfully",
            "booking_reference": booking_reference,
            "status": "cancelled",
        }

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not cancel booking",
        ) from exc


# ---------------------------------------------------------
# Download booking PDF
# ---------------------------------------------------------


# Add these imports to your existing main.py if they are not already present.

@app.get("/api/bookings/{booking_reference}/receipt")
def download_booking_receipt(
    booking_reference: str,
    email: EmailStr,
    db: Session = Depends(get_db),
):
    query = text("""
        SELECT
            b.id,
            b.booking_reference,
            b.status,
            b.check_in,
            b.check_out,
            b.total_amount,
            b.created_at,
            b.adults,
            b.children,
            c.name AS customer_name,
            c.email AS customer_email,
            c.phone AS customer_phone,
            rc.name AS room_category,
            rc.price_per_night
        FROM bookings b
        JOIN customers c ON c.id = b.customer_id
        JOIN room_categories rc ON rc.id = b.room_category_id
        WHERE b.booking_reference = :reference
          AND LOWER(c.email) = LOWER(:email)
    """)

    booking = db.execute(
        query,
        {
            "reference": booking_reference,
            "email": str(email),
        },
    ).mappings().first()

    if booking is None:
        raise HTTPException(
            status_code=404,
            detail="Booking not found",
        )

    # Resolve assets relative to this Python file, not the current terminal directory.
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "assets"
    logo_path = assets_dir / "al_hayat_logo.jpg"
    font_path = assets_dir / "DejaVuSans.ttf"

    if not logo_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Hotel logo file not found: {logo_path}",
        )

    if not font_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Arabic font file not found: {font_path}",
        )

    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", str(font_path)))

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title=f"Al Hayat Hotel Reservation {booking_reference}",
    )

    styles = getSampleStyleSheet()

    hotel_title = ParagraphStyle(
        "HotelHeader",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#A88A19"),
        spaceAfter=2,
    )

    arabic_style = ParagraphStyle(
        "HotelArabic",
        parent=styles["Normal"],
        fontName="DejaVu",
        fontSize=14,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
        spaceAfter=12,
    )

    normal = ParagraphStyle(
        "ReservationText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#333333"),
    )

    small = ParagraphStyle(
        "SmallText",
        parent=normal,
        fontSize=8,
        leading=10,
    )

    section_style = ParagraphStyle(
        "SectionBar",
        parent=normal,
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.white,
    )

    story = []

    # Hotel logo and bilingual hotel name.
    logo = PDFImage(
        str(logo_path),
        width=62 * mm,
        height=24 * mm,
        kind="proportional",
    )
    logo.hAlign = "CENTER"
    story.append(logo)
    story.append(Spacer(1, 5))
    story.append(Paragraph("Al Hayat Hotel", hotel_title))

    arabic_name = get_display(
        arabic_reshaper.reshape("فندق الحياة")
    )
    story.append(Paragraph(arabic_name, arabic_style))

    # Hotel contact information and VAT number.
    contact_rows = [[
        Paragraph(
            "<b>Al Hayat Hotel LLC</b><br/>"
            "PO Box 79193, Sharjah<br/>"
            "Tel: +971 6 5066 222<br/>"
            "Website: www.alhayat-hotel.com<br/>"
            "Email: reservation@alhayat-hotel.com",
            normal,
        ),
        Paragraph(
            "<b>VAT Registration:</b><br/>100272615400003",
            normal,
        ),
    ]]

    contact_table = Table(
        contact_rows,
        colWidths=[105 * mm, 65 * mm],
    )
    contact_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(contact_table)
    story.append(Spacer(1, 5))

    confirmation_date = (
        booking["created_at"].strftime("%d/%m/%Y")
        if booking["created_at"]
        else date.today().strftime("%d/%m/%Y")
    )

    reference_bar = Table(
        [[
            Paragraph(
                "<b>Reservation Confirmation No: "
                f"{escape(str(booking['booking_reference']))}</b>",
                normal,
            ),
            Paragraph(f"<b>Date: {confirmation_date}</b>", normal),
        ]],
        colWidths=[110 * mm, 60 * mm],
    )
    reference_bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#D9D9D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(reference_bar)
    story.append(Spacer(1, 12))

    guest_name = escape(str(booking["customer_name"]))
    status = str(booking["status"]).lower()

    if status == "confirmed":
        greeting = (
            f"Dear <b>{guest_name}</b>,<br/>"
            "Thank you for choosing Al Hayat Hotel. "
            "We are pleased to confirm your reservation. "
            "Kindly review the reservation details below."
        )
    elif status == "cancelled":
        greeting = (
            f"Dear <b>{guest_name}</b>,<br/>"
            "This reservation has been cancelled. "
            "Please refer to the details below."
        )
    else:
        greeting = (
            f"Dear <b>{guest_name}</b>,<br/>"
            "Thank you for choosing Al Hayat Hotel. "
            "Your reservation request has been recorded and "
            "is awaiting confirmation by the hotel. "
            "Please review the details below."
        )

    story.append(Paragraph(greeting, normal))
    story.append(Spacer(1, 12))

    nights = (booking["check_out"] - booking["check_in"]).days
    total_guests = booking["adults"] + booking["children"]

    rows = [
        [Paragraph("<b>Guest Name</b>", normal), Paragraph(guest_name, normal)],
        [
            Paragraph("<b>Email</b>", normal),
            Paragraph(escape(str(booking["customer_email"])), normal),
        ],
        [
            Paragraph("<b>Phone</b>", normal),
            Paragraph(escape(str(booking["customer_phone"] or "")), normal),
        ],
        [
            Paragraph("<b>Arrival Date</b>", normal),
            Paragraph(booking["check_in"].strftime("%d/%m/%Y"), normal),
        ],
        [
            Paragraph("<b>Departure Date</b>", normal),
            Paragraph(booking["check_out"].strftime("%d/%m/%Y"), normal),
        ],
        [
            Paragraph("<b>No. of Guests</b>", normal),
            Paragraph(str(total_guests), normal),
        ],
        [
            Paragraph("<b>Adults / Children</b>", normal),
            Paragraph(
                f"{booking['adults']} adult(s), {booking['children']} child(ren)",
                normal,
            ),
        ],
        [
            Paragraph("<b>Room Type</b>", normal),
            Paragraph(escape(str(booking["room_category"])), normal),
        ],
        [
            Paragraph("<b>No. of Nights</b>", normal),
            Paragraph(str(nights), normal),
        ],
        [
            Paragraph("<b>Rate per Night</b>", normal),
            Paragraph(f"AED {booking['price_per_night']:,.2f}", normal),
        ],
        [
            Paragraph("<b>Total Room Amount</b>", normal),
            Paragraph(f"AED {booking['total_amount']:,.2f}", normal),
        ],
        [
            Paragraph("<b>Billing Instructions</b>", normal),
            Paragraph("Direct Payment", normal),
        ],
        [
            Paragraph("<b>Booking Status</b>", normal),
            Paragraph(escape(status.upper()), normal),
        ],
    ]

    details_table = Table(
        rows,
        colWidths=[55 * mm, 115 * mm],
        hAlign="LEFT",
    )
    details_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#DDDDDD")),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 14))

    policy_heading = Table(
        [[Paragraph("IMPORTANT RESERVATION INFORMATION", section_style)]],
        colWidths=[170 * mm],
    )
    policy_heading.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#777777")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(policy_heading)
    story.append(Spacer(1, 5))

    # Verify all policies and fees with the hotel before using this PDF live.
    policies = [
        "Reservations are subject to hotel confirmation. Please contact "
        "the reservation department regarding guarantees and cancellation conditions.",
        "The supplied sample states that rates include a 10% municipality fee "
        "and a 10% service charge, and are subject to 5% VAT. Confirm applicable "
        "charges with the hotel before collecting payment.",
        "Customer preferences and special requests are subject to availability. "
        "Pets are not allowed.",
        "All guests must check in together and present an original passport "
        "or Emirates ID.",
        "Standard check-in is at 2:00 PM and check-out is at 12:00 PM. Early "
        "check-in and late check-out are subject to availability and may incur "
        "additional charges.",
        "For booking enquiries, contact reservation@alhayat-hotel.com or visit "
        "www.alhayat-hotel.com.",
    ]

    for policy in policies:
        story.append(Paragraph(f"&bull;&nbsp; {escape(policy)}", small))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Best Regards,<br/>"
            "<b>Reservation Department</b><br/>"
            "Al Hayat Hotel Sharjah",
            normal,
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "This document is a booking summary and is not proof of payment. "
            "Payment has not been processed through this booking form.",
            small,
        )
    )

    document.build(story)
    buffer.seek(0)

    safe_reference = "".join(
        char for char in booking_reference
        if char.isalnum() or char in "-_"
    )

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="booking-{safe_reference}.pdf"'
            ),
            "Cache-Control": "no-store",
        },
    )

