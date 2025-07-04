import React from 'react';
import { getInitials } from '@/utils/app/helper';

export const UserAvatar = ({ src = '', height = 30, width = 30 }) => {
    const onError = (event: { target: { src: string; }; }) => {
        if (!event.target.src.includes('/avatar-default.png')) {
            event.target.src = '/avatar-default.png';
        }
    };

    return (
        <img 
            src={src || '/avatar-default.png'} 
            alt="user-avatar" 
            width={width}
            height={height}
            title="user-avatar"
            className="rounded-full max-w-full h-auto border"
            onError={onError}
        />
    );
};